'''****************
* tss_graph_builder.py
*
* Parse an input TSS graph spec and generate
* the extended specification, including rungs,
* windows, and nodes. Generate and merge partial
* windows. Perform extensive assertion checking
* of the input spec and the generated output. 
'''
import os, copy
from ark import Ark, File, _ark
import numpy as np
from scipy.interpolate import interp2d 
import itertools as it
from collections import Counter, defaultdict, namedtuple

from .graph_validation import check_two_windows_every_rung
from .interpolate import *
from .util import first_different_index, merge_dicts, param_equiv, tup_replace

__all__ = ['GraphBuilder']

def flatten_neighbors_dimensional(dn):
    return [neighbor for neighbor_pair in dn for neighbor in neighbor_pair]


def get_neighbors_dimensional(edge_id, index, rung_loc_to_index):
    neighbors_dimensional = []

    # Generate neighbors in each dimension
    for dim, value in enumerate(index):
        before_neighbor = (edge_id, tup_replace(index, dim, value-1))
        self = (edge_id, index)
        after_neighbor = (edge_id, tup_replace(index, dim, value+1))

        # Filter out invalid neighbors (beyond array bounds)
        if before_neighbor not in rung_loc_to_index:
            before_neighbor = self
        if after_neighbor not in rung_loc_to_index:
            after_neighbor = self

        neighbors_dimensional.append(
            [rung_loc_to_index[before_neighbor], rung_loc_to_index[after_neighbor]])

    return neighbors_dimensional


# Compute the line of partial windows within an edge,
# in the order [pw1, ..., pw2]
def partial_windows_between(edge_id, pw1, pw2, all_partial_windows):
    # only one varying index for a line merge
    varying_index = first_different_index(pw1.start_rungs, pw2.start_rungs)
    reverse_order = pw1.start_rungs[varying_index] > pw2.start_rungs[varying_index]

    shared_pms = set(pw1.pms) & set(pw2.pms)
    # Matching partial windows have at least the same partial memberships
    # (have same boundaries) as the two end pws
    pws = [pw for pw in all_partial_windows[edge_id] if shared_pms <= set(pw.pms)]
    sorted_pws = sorted(pws, key = lambda pw: pw.start_rungs[varying_index], reverse = reverse_order)

    return (varying_index, reverse_order, sorted_pws)

def get_merger_rungs(merge_item):
    '''Given a merge item containing a partial window and merge dimension,
        compute the line of rungs along the merge axis'''
    # a. Obtain merge perpendicular dimension if it exists
    mpd = merge_item.merge_perpendicular_dimension

    # b. Get coords of one merger rung based on partial memberships
    base_coords = merge_item.partial_window.start_rungs[:]

    for pm in merge_item.partial_window.pms:
        if pm.start_or_end == 1 and pm.dimension != mpd:
            # at end
            # the coord of that rung is (start + (size - 1))
            # it is not (start + size):
            # if start = 0, size = 8, then end is 7. 
            # problems with this line usually relate to an incorrect start location
            # perhaps caused by an odd number of rungs in the partial window
            base_coords[pm.dimension] = base_coords[pm.dimension] + (merge_item.partial_window.window_sizes[pm.dimension] - 1)

    # c. If applicable, extend upward along the merge perpendicular dimension
    if mpd == None:
        # point merge, only one rung from each edge
        return [base_coords]
    else:
        # generate the line of rungs involved in each edge 
        ndim = len(merge_item.partial_window.window_sizes)
        nrungs = merge_item.partial_window.window_sizes[mpd]

        merge_rungs = [] * nrungs

        # Ensure that if this pw is at a corner along the merge parallel dimension,
        # our output rung order begins with the corner rung and that we generate
        # rungs downward instead of upward. 

        # partial membership in the merge propagation direction
        partial_membership_mpd_start_or_end = next((pm.start_or_end for pm in merge_item.partial_window.pms if pm.dimension == mpd), None)

        for i in range(nrungs):
            merge_rungs.append(base_coords[:])
            merge_rungs[i][mpd] = base_coords[mpd] + i

        if partial_membership_mpd_start_or_end == 1 or \
            (merge_item.reverse_rungs and not partial_membership_mpd_start_or_end == 0):
            merge_rungs.reverse()

        return merge_rungs

class TSSRung(object):
    def __init__(self):
        self.values = {}
        self.neighbors_dimensional = []

        self.edge_id = None
        self.flat_index = None
        self.location = None

        self.edge_shape = None

    def __str__(self):
        return "<TSSRung, {}>".format(self.values)

    def __repr__(self):
        return str(self)

    def compute_volume(self):
        loc = np.array(self.location)
        at_edge_of_dimensions = np.logical_or(loc == 0, loc == (self.edge_shape - 1))
        n_edge_dims = np.sum(at_edge_of_dimensions)

        volume = (0.5 ** n_edge_dims)
        return volume

    def cerealize(self):
        return {'parameters': self.values,
                'neighbors_dimensional': self.neighbors_dimensional,
                'flat_index': self.flat_index,
                'edge_id': self.edge_id,
                'location': self.location,
                'volume': self.compute_volume()}

class TSSWindow1D(object):
    '''Represents an intermediate step in graph generation. TSSWindow1Ds
        are generated along each edge dimension separately, and then the
        Cartesian product across all the dimensions is taken to generate
        the final set of TSSWindows. Even in a 1-D graph this class is never
        an end product.'''
    def __init__(self, start_rung, window_size):
        self.start_rung = start_rung
        self.window_size = window_size
        self.partial_memberships = []
        self.window_coordinates = []

    def __str__(self):
        return "<TSSWindow1D, {} {}>".format(self.start_rung, self.window_size)

    def __repr__(self):
        return str(self)

    # internal object, so no cerealize required


class TSSWindow(object):
    '''Represents an n-dimensional window contained entirely within one edge.
        Its extent in each dimension is [start_rungs[i], start_rungs[i] + window_sizes[i]]
        Its partial memberships (pms) are used in cross-edge merge processing.'''
    def __init__(self, windows_1d):
        self.start_rungs = [w.start_rung for w in windows_1d]
        self.window_sizes = [w.window_size for w in windows_1d]
        self.pms = sorted(list(it.chain(*[w.partial_memberships for w in windows_1d])), key=lambda pm: pm.dimension)
        self.dimensions = len(windows_1d)

    def __str__(self):
        return "<TSSWindow, {} {}>".format(self.start_rungs, self.window_sizes)

    def __repr__(self):
        return str(self)

    def is_partial(self):
        """If this is a partial window, then our list of partial memberships is nonempty"""
        return len(self.pms) > 0

    def at_corner(self, xyz):
        return len(self.pms) == len(xyz) and all([xyz[i] == self.pms[i].start_or_end for i in range(len(xyz))])

    def cerealize(self, edge_id, rung_eindex_to_id):
        # rung_eindex_to_id: dictionary mapping (edge_id, rung_coords) to rung_flat_index
        # This edge is fully antitheticable

        # n-dim array whose contents are rung indices in the output array
        rung_array = np.empty(self.window_sizes, dtype=int)

        for rung_delta_index in np.ndindex(tuple(self.window_sizes)):
            rung_loc = tuple(np.array(self.start_rungs) + rung_delta_index)
            rung_index = rung_eindex_to_id[(edge_id, rung_loc)]

            rung_array[rung_delta_index] = rung_index

        return {'rung_set': [rung_array],
                '__edge_id': edge_id,
                '__partial_memberships': self.pms}

class TSSMultiEdgeWindow(object):
    '''Contains several single-edge windows, representing a merger across edges.'''
    def __init__(self, merge_proposals):
        self.mps = merge_proposals # [TSSMergeItem, ...]

    def is_antithetic(self):
        '''Merge is antithetic if it's a line merge, which would mean
            every partial window joined has a perpendicular dimension.'''
        return all([w.merge_perpendicular_dimension != None for w in self.mps.values()])

    def __str__(self):
        return "<TSSMultiEdgeWindow, {}>".format(self.mps)

    def __repr__(self):
        return str(self)

    def cerealize(self, rung_eindex_to_id):
        all_cerealized = []

        for edge_id, merge_item in self.mps.items():
            item_cerealized = merge_item.partial_window.cerealize(edge_id, rung_eindex_to_id)
            all_cerealized.append(item_cerealized)

        merged_data = merge_dicts(all_cerealized) # combine all the individual window dicts

        if self.is_antithetic():
            # remove the outermost [] (indicating antitheticability) from each
            # partial window to be merged
            merged_data['rung_set'] = np.array([item[0] for item in merged_data['rung_set']])
        else:
            # only independent moves allowed, so flatten all rungs so output is a 1-D rung list
            flattened_rungs = np.concatenate([np.array(item).flatten() for item in merged_data['rung_set']])
            merged_data['rung_set'] = flattened_rungs

        return merged_data

class TSSNode(object):
    def __init__(self, name):
        if not isinstance(name, str):
            raise TypeError(name)
        self.name = name
        self.node_locations = [] # [TSSNodeLocation, ...]
        self.rung_set = {} # {edge_id: rung coords, ...}
        self.flat_rung_ids = [] # [flat_rung_index, ...]

    def cerealize(self, rung_eindex_to_id):
        rung_set = []
        for edge_id, rung_index in self.rung_set.items():
            rung_set.append(rung_eindex_to_id[(edge_id, rung_index)])

        return {'name': self.name, 'rung_set': rung_set}

    def __str__(self):
        return "<TSSNode, {} {} {}>".format(self.name, self.node_locations, self.rung_set)

    def __repr__(self):
        return str(self)

    
TSSNodeLocation = namedtuple('TSSNodeLocation', ['edge_id', 'location'])
TSSPartialMembership = namedtuple('TSSPartialMembership', ['dimension', 'start_or_end'])
TSSMergeItem = namedtuple('TSSMergeItem', ['partial_window', 'merge_perpendicular_dimension', 'reverse_rungs'])
TSSEdge = namedtuple('TSSEdge', ['node_set'])

class GraphBuilder:
    '''Represents a complete TSS graph.'''
    def __init__(self, input_ark=None):
        '''Initialize from ark or ark-like dict structure of edges.

        Args:
             ark_graph (ark/dict): The equivalent graph ark from integrator.times_square in an Anton2 ark.'''

        self.rungs = {} # {edge_id: np array of TSSRung objects, ...}
        self.windows = {} # {edge_id: [window, ...], ...}
        self.partial_windows = {} # partial windows for possible merger with neighboring edge partial windows
        self.merged_windows = [] # [TSSMultiEdgeWindow, ...]
        self.nodes = {} # {node_name: [rungs included], ...}
        self.rung_locs = {} # {(edge_id, rung_index): flattened_rung_index, ...}

        self.edges = [] # [TSSEdge, ...]
        self.blocks = [] # [[flattened_rung_index, ...], ...]

        if input_ark:
            self.build(input_ark)

    def _insert_rung_neighbors(self):
        '''After all merges have been processed, inform rungs of their neighbors.'''

        # 1. Enumerate all the rungs to construct a flattened 1-D index
        # mapping (edge_id, rung_position) -> flat_index
        flat_index = 0
        for edge_id, rungs in self.rungs.items():
            for index, rung in np.ndenumerate(rungs):
                self.rung_locs[(edge_id, index)] = flat_index
                rung.edge_id = edge_id
                rung.flat_index = flat_index

                flat_index = flat_index + 1

        # 2. Figure out rung dimensional neighbors
        for edge_id, rungs in self.rungs.items():
            for index, rung in np.ndenumerate(rungs):
                 rung.neighbors_dimensional = get_neighbors_dimensional(edge_id, index, self.rung_locs)

        # 3. Check for boundary rungs with extended neighbors

        # get all rungs belonging to any nodes (that is, on a boundary)
        boundary_rungs_to_nodename = {}
        nodename_to_rungs = defaultdict(list)

        for node_name, node in self.nodes.items():
            for rung_edge_id, rung_index in node.rung_set.items():
                # Determine the node rung's flat index
                flat_index = self.rung_locs[(rung_edge_id, rung_index)]

                # Store maps of flat index -> node and node -> flat index
                boundary_rungs_to_nodename[flat_index] = node_name
                node.flat_rung_ids.append(flat_index)

    def cerealize(self, filename=None):
        '''Convert internal builder graph representation to final TSS ark object. 
            Perform sanity checks on generated graph.

            If filename is supplied, ark is written there. Otherwise an ark object
            is returned.'''
        # 1. Prepare to output rungs. 
        rungs_in_order = np.empty(len(self.rung_locs), dtype=object)
        for (edge_id, rung_index), rung_flat_index in self.rung_locs.items():
            rungs_in_order[rung_flat_index] = self.rungs[edge_id][rung_index]

        rungs_cerealized = [rung.cerealize() for rung in rungs_in_order]

        # 2. Serialize windows 
        windows_cerealized = []
        for edge_id, window_set in self.windows.items():
            for window in window_set:
                windows_cerealized.append(window.cerealize(edge_id, self.rung_locs))
        
        for edge_id, window_set in self.partial_windows.items():
            for window in window_set:
                windows_cerealized.append(window.cerealize(edge_id, self.rung_locs))


        for window in self.merged_windows:
            windows_cerealized.append(window.cerealize(self.rung_locs))

        # 3. Serialize nodes
        nodes_cerealized = [n.cerealize(self.rung_locs) for n in sorted(self.nodes.values(), key=lambda n: n.name)]

        # 4. Serialize blocks
        blocks_cerealized = [{'rung_set': s} for s in self.blocks]

        # 5. Serialize edges
        edges_cerealized = [{'node_set': e.node_set} for e in self.edges]

        # 5. Generate ark
        d = {'graph': {
                'rungs': rungs_cerealized,
                'windows': windows_cerealized,
                'nodes': nodes_cerealized,
                'blocks': blocks_cerealized,
                'edges': edges_cerealized,
                '__tss_graph_ark_version': 1,
                '__input_ark': self.input_ark
                }
            }

        ark = Ark(d)

        if filename:
            ark.save(filename, open_tables=True)
        else:
            return ark

    def add_schedule_ark(self, edge_id, sched):
        '''Add a parameter schedule to an existing edge from an ark spec. 
            
            Args:
                edge_id: ID of edge. 
                sched: ark block containing schedule information.
        '''
        sched = copy.deepcopy(sched)
    
        if 'interpolation' not in sched:
            sched['interpolation'] = 'linear'
        if 'dimension' not in sched:
            sched['dimension'] = 0
        if 'degree' not in sched:
            sched['degree'] = 2
        if 'bias' not in sched:
            sched['bias'] = None
        if 'number_of_windows' in sched:
            warnings.warn("'number_of_windows' key is ignored in schedule construction, use 'window_size' instead'", DeprecationWarning)   
        assert(isinstance(sched['dimension'], int) or isinstance(sched['dimension'], list)), "schedule 'dimension' key must be int or list"

        if isinstance(sched['dimension'], list):
            self.add_schedule_2d(edge_id, sched['group_name'], sched['interpolation'], sched['bounds'], sched['bias'], sched['dimension'])
        else:
            self.add_schedule(edge_id, sched['group_name'], sched['interpolation'], sched['bounds'], sched['degree'], sched['bias'], sched['dimension'])

    def add_schedule(self, edge_id, group_name, interpolation, bounds, degree=2, bias=None, dim=0):
        '''Add a parameter schedule to a given edge, once that edge is created.

            Args:
                edge_id: the id provided to add_edge along which this schedule is to be placed

                group_name: name of the tempered parameter (str)

                interpolation: 'linear', 'piecewise', 'polynomial', 'geometric', 'geometric_increment', or 'explicit' (see TSS documentation)

                bounds: [bound1, bound2] (floats)

                degree: int (only for 'polynomial')

                bias: float (only for geometric_increment)

                dim: for multi-d edges, dimension along which this schedule should be added (int)

        '''

        assert (edge_id in self.rungs), "edge_id {} has not been added".format(edge_id)

        number_of_rungs = self.rungs[edge_id].shape[dim]

        interpolation_choices = {'linear': interpolate_linear,
                                'piecewise': interpolate_piecewise,
                                'polynomial': interpolate_polynomial,
                                'geometric': interpolate_geometric,
                                'geometric_increment': interpolate_geometric_increment,
                                'explicit': interpolate_explicit}
        assert (interpolation in interpolation_choices), "interpolation type must be one of {}".format(list(interpolation_choices.keys()))

        if interpolation == 'polynomial':
            bounds = bounds + [degree]
        if interpolation == 'geometric_increment':
            bounds = bounds + [bias]

        interp_fn = interpolation_choices[interpolation]
        rung_values = interp_fn(bounds, number_of_rungs)

        # And store them in the rung objects
        for i, v in enumerate(rung_values):
            # Our rungs are n-dimensional and we want to modify all the entries
            # in the rung hyperplane at value i of dimension dim
            for rung in np.rollaxis(self.rungs[edge_id], dim)[i:i+1].flat:
                rung.values[group_name] = v


    def add_schedule_2d(self, edge_id, group_name, interpolation, bounds, bias=None, dim=[0,1]):
        '''Add a parameter schedule to a given edge, once that edge is created. Performs 2-D interpolation. 

            Args:
                edge_id: the id provided to add_edge along which this schedule is to be placed

                group_name: name of the tempered parameter (str)

                interpolation: currently 'linear' only is supported (see TSS documentation)

                bounds: [bound1, bound2] (floats)

                bias: float (only for geometric_increment)

                dim: 2 dimensions along which this schedule should be added [dim0, dim1] (ints)

        '''

        assert (edge_id in self.rungs), "edge_id {} has not been added".format(edge_id)

        number_of_rungs = np.array(self.rungs[edge_id].shape)[dim]

        assert (interpolation == "linear"), "interpolation type for 2-D schedule must be linear"

        grid_interp = interp2d([0, 0, number_of_rungs[0] - 1, number_of_rungs[0] - 1], 
                               [0, number_of_rungs[1] - 1, 0, number_of_rungs[1] - 1],
                                bounds)

        rung_values = grid_interp(np.arange(number_of_rungs[0]), np.arange(number_of_rungs[1])).T

        # And store them in the rung objects
        for i, v in np.ndenumerate(rung_values):
            # Our rungs are n-dimensional and we want to modify all the entries
            # in the rung hyperplane at value i of dimension dim
            for rung in np.array(np.moveaxis(self.rungs[edge_id], dim, (0, 1))[i]).flat:
                rung.values[group_name] = v


    def add_edge_ark(self, ark_edge):
        '''Add an edge from an ark spec.

            Args:
                ark_edge: the ark block containing edge info.

            Returns:
                edge_id: the ID of the newly added edge. '''
        ark_edge = copy.deepcopy(ark_edge)

        listify = lambda x: [x] if type(x) == int else x

        if 'dimensions' not in ark_edge:
            ark_edge['dimensions'] = 1

        if 'primary_window_tiling_only' not in ark_edge:
            ark_edge['primary_window_tiling_only'] = False

        assert('number_of_rungs' in ark_edge), "number_of_rungs missing in edge"
        assert('window_size' in ark_edge), "window_size (count of rungs per window) missing in edge"

        ark_edge['number_of_rungs'] = listify(ark_edge['number_of_rungs'])
        ark_edge['window_size'] = listify(ark_edge['window_size'])

        #print("*****")
        #print(ark_edge['number_of_rungs'])
        #print(ark_edge['dimensions'])

        return self.add_edge(ark_edge['nodes'], ark_edge['number_of_rungs'], ark_edge['window_size'], ark_edge['dimensions'], ark_edge['primary_window_tiling_only'])


    def add_edge(self, node_array, number_of_rungs, window_size, dimensions=1, primary_window_tiling_only=False):
        '''Add a new edge to the graph.

            Args: 
                node_array: ["name_0", "name_1", ...] with appropriate nesting for a d-dimensional array. So, for instance, a 1-d edge is always [n1, n2] and a 2-d edge is [[n1, n2], [n3, n4]]

                number_of_rungs: [rungs in d1 (int), rungs in d2, ...]

                window_size: [size in d1 (int), size in d2, ...]

                dimensions: int

                primary_window_tiling_only: boolean. Expert setting only: if set to True, does not generate the second (overlapping) window tiling for this edge. That tiling is necessary for cross-edge window merges, so only recommended on a single edge. 

            Returns:
                edge_id: integer id of the new edge. Required for add_schedule
        '''
        assert(len(number_of_rungs) == dimensions), "Length of number_of_rungs must match edge dimension"
        assert(len(window_size) == dimensions), "Length of window_size must match edge dimension"

        node_array = np.array(node_array, dtype=np.dtype('U50'))
        assert(node_array.shape == tuple(np.repeat(2, dimensions))), "nodes must be a d-dimensional array with 2 elements on each segment; instead it has shape {}".format(node_array.shape)

        c = Counter(node_array.flat)
        for item, count in c.items():
            assert(count == 1 or item == "_"), "Vertex name \"{}\" is repeated within an edge and is not \"_\"".format(item)


        # Create and initialize n-dimensional rung array
        edge_rungs = np.empty(number_of_rungs, dtype=object)
        edge_rungs.flat = [TSSRung() for _ in edge_rungs.flat]

        # Tell each rung where it is
        for index, rung in np.ndenumerate(edge_rungs):
            rung.location = list(index) # list is more cerealizable as ark 
            rung.edge_shape = np.array(edge_rungs.shape)

        # Create a new edge_id
        edge_id = len(self.rungs)

        self.rungs[edge_id] = edge_rungs

        # After rung creation, add all the vertex rungs to their respective labeled nodes
        for dimensional_position, node_name in np.ndenumerate(node_array):

            # Nodes with input name "_" are anonymous and given internal names. 
            # They cannot be used to join edges. 
            node_array[dimensional_position] = node_name if node_name != "_" \
                else "unnamed_node_{}_{}".format(edge_id, dimensional_position)

        # After rung creation, add all the vertex rungs to their respective labeled nodes
        for dimensional_position, node_name in np.ndenumerate(node_array):
            rung_position = []

            # Add to dict
            if node_name not in self.nodes:
                self.nodes[node_name] = TSSNode(node_name)

            # Compute the rung index corresponding to this node location
            # tuple for numpy indexing
            zeros = [0] * dimensions
            end_positions = np.array(number_of_rungs) - 1
            rung_position = tuple(np.choose(dimensional_position, [zeros, end_positions]))
        
            # Tell the node where it is in this edge and what rung it includes
            self.nodes[node_name].node_locations.append(TSSNodeLocation(edge_id, dimensional_position))
            self.nodes[node_name].rung_set[edge_id] = rung_position

        # Store edge's node list
        self.edges.append(TSSEdge(node_array))

        # Generate windows
        (windows, partial_windows) = self._generate_windows(number_of_rungs, window_size, not primary_window_tiling_only)
        self.windows[edge_id] = windows
        self.partial_windows[edge_id] = partial_windows

        return edge_id

    def _generate_windows(self, number_of_rungs, window_size, generate_overlapping=True):
        '''For a single edge, construct n-dimensional window arrays. Each window has the
            same number of dimensions as the edge, and the windows tile the edge like a
            checkerboard. 
        
            Args:
                number_of_rungs: list with number of rungs in each dimension
                window_size: list with number of rungs per window in each dimension. 

         - Regular tiling: exactly lines up with edge boundaries
         - Overlapping tiling: offset in all dimensions, including all partial windows (for merger with neighboring edges). Produced in addition to regular tiling. 

        This function is also used to generate miniblocks, since it really just produces a tiling
        of the edge. 
        '''

        windows_1d_regulartiling = [] # for each dimension, window boundaries within that dimension
        windows_1d_overlappingtiling = [] # for each dimension, window boundaries within that dimension

        # Iterate through dimensions and generate window sizes along each dim
        for i, ws in enumerate(window_size):
            assert ws == int(ws), 'window size must be an integer: got {}'.format(ws)
            ws = int(ws)
            nr = number_of_rungs[i]

            if generate_overlapping:
                assert (ws / 2.0 == ws // 2), "window size must be even in dimension {} so that half-size partial windows are of integer size".format(i)
            assert (nr % ws)  == 0, "number of rungs must be multiple of window size in dimension {}".format(i)

            # Nonoverlapping windows
            nnonoverlapping = nr // ws # other windows overlap with these

            rung_starts = np.arange(nnonoverlapping, dtype=int) * ws
            windows_dim_i_regulartiling = [TSSWindow1D(rs, ws) for rs in rung_starts]

            # Tiling    
            windows_1d_regulartiling.append(windows_dim_i_regulartiling)

            if generate_overlapping:
                rung_starts_offset = (np.arange(nnonoverlapping - 1, dtype=int) * ws) + (ws // 2) # overlapping windows
                windows_dim_i_overlappingtiling = [TSSWindow1D(rs, ws) for rs in rung_starts_offset]

                # Generate partial windows at beginning and end
                p1 = (TSSWindow1D(0, ws//2))
                p2 = (TSSWindow1D(rung_starts[-1] + ws//2, ws//2))

                p1.partial_memberships.append(TSSPartialMembership(i, 0))
                p2.partial_memberships.append(TSSPartialMembership(i, 1))

                windows_dim_i_overlappingtiling.extend([p1, p2])

                windows_1d_overlappingtiling.append(windows_dim_i_overlappingtiling)
        

        # Get the Cartesian product across all the dimensions of the rung start indices to produce the n-dim windows

        # windows_1d_regulartiling: [[a, b, c], [d, e, f], ]
        # after meshgrid: [a, b, c, a, b, c, a, b, c, ...], [d, d, d, e, e, e, f, f, f, ...], ...]
        # after izip: [(a, d, ...), (b, d, ...), (c, d, ...), ...]
        # where all the letters are TSSWindow1Ds
        window_1d_groups = it.chain(zip(*[l.flat for l in np.meshgrid(*windows_1d_regulartiling)]),
                                    zip(*[l.flat for l in np.meshgrid(*windows_1d_overlappingtiling)]))
        
        windows = []
        partial_windows = []

        for window_1d_g in window_1d_groups:
            # We have a set of 1-D windows to combine to produce a hypercube window
            combined_w = TSSWindow(window_1d_g)

            if combined_w.is_partial():
                partial_windows.append(combined_w)
            else:
                windows.append(combined_w)


        return (windows, partial_windows)

    

    def _propose_node_merges(self):
        """After all edges are added, examine the set of nodes to see which edges
            should be merged."""
        merge_proposals = []
        partial_window_at_corner = lambda edge_id, loc: next((pw for pw in self.partial_windows[edge_id] if pw.at_corner(loc)), None)

        for n in self.nodes.values():
            if len(n.node_locations) > 1:
                # belongs to more than one edge
                merge_proposal = {e.edge_id: 
                                    TSSMergeItem(partial_window_at_corner(e.edge_id, e.location),
                                            None, # mpd and reverse will be populated later
                                            None)
                                  for e in n.node_locations}

                # delete entries from the merge proposal if no partial window found (usually a result of primary_window_tiling_only)
                merge_proposal = {k: v for k, v in merge_proposal.items() if v.partial_window}

                if len(merge_proposal) > 1:
                    merge_proposals.append(merge_proposal)

        return merge_proposals

    def _propagate_fixed_parameters(self, merge_nodes):
        """Given a set of merge nodes, identify parameters that vary along an edge A but
            not along its neighbor edge B, so the final values from edge A should be propagated
            across all rungs in B."""

        any_updated = False

        for n in merge_nodes.values():
            for edge1, rung1_loc in n.rung_set.items():
                for edge2, rung2_loc in n.rung_set.items():
                    if edge1 == edge2:
                        continue

                    rung1 = self.rungs[edge1][rung1_loc]
                    rung2 = self.rungs[edge2][rung2_loc]

                    propagate_param_names = list(set(rung1.values.keys()) - set(rung2.values.keys()))
                    propagate_params = {k: rung1.values[k] for k in propagate_param_names}
                    
                    if len(propagate_params):
                        any_updated = True
                        for e2_rung in self.rungs[edge2].flat:
                            e2_rung.values.update(propagate_params)

        if any_updated:
            self._propagate_fixed_parameters(merge_nodes)
            # repeat recursively until all propagation has occurred

    def _extend_merges(self, merge_proposals, partial_windows):
        """Analyze proposed node merges and decide whether those merges should take place at a point
            (one partial window from each edge involved) and/or along a line (a 1-D set of
            windows from each edge. 

            Edges of any dimension are supported. However, this function only supports merging
            them at points or lines, not along planes or hyperplanes. That feature would
            require more sophisticated merge extension and proposal logic.

            merge_proposals: [{edge_id: TSSMergeItem, ...}, ...]
            partial_windows: {edge_id: [TSSWindow, ...], ...}

            return: extended_merge_proposals
        """

        interpolated_merge_proposals = []

        # Detect whether to merge along a line by looking for pairs of merge proposals involving the same edges
        for a, mp1 in enumerate(merge_proposals):
            for b, mp2 in enumerate(merge_proposals[a+1:]):
                overlap = list(set(mp1.keys()) & set(mp2.keys()))

                #------------Change this part to support n-d edge mergers----
                #------------the hardest part may be simply ordering the partial windows correctly
                if len(overlap) <= 1:
                    # only one edge shared: not a merge we care about
                    continue
                if any([mi.partial_window.dimensions < 2 for mi in mp1.values()]) or any([mi.partial_window.dimensions < 2 for mi in mp2.values()]):
                    # do not perform line merges on 1-D windows (should be stitched at ends only)
                    continue

                # Identify partial windows along the shared axis of the shared edges
                # and construct merge proposals for them too

                # For each edge involved in the merger, get the index of the coord that
                # traces its boundary and the list of partial windows along the boundary
                # in the correct order to be joined
                (varying_indices, reverse_rungs, partial_window_lists) = zip(*[partial_windows_between(edge_id, mp1[edge_id].partial_window, mp2[edge_id].partial_window, self.partial_windows) for edge_id in overlap])

                npw = [len(s) for s in partial_window_lists]
                assert(len(set(npw)) == 1), "Partial window merger failed because edges along merge line have different numbers of windows in this dimension"

                # Create groups of partial windows to be joined
                merge_sets = list(zip(*partial_window_lists))

                assert(len(merge_sets) >= 2), "Identified two nodes for line join but window interpolation failed"

                # For each edge in the merge, process its first and last partial windows
                # which are in the two existing merge proposals used to generate the line join
                for i, edge_id in enumerate(overlap):
                    mp1[edge_id] = mp1[edge_id]._replace(merge_perpendicular_dimension=varying_indices[i], reverse_rungs=reverse_rungs[i])
                    mp2[edge_id] = mp2[edge_id]._replace(merge_perpendicular_dimension=varying_indices[i], reverse_rungs=reverse_rungs[i])
            
                # No more information needed for mp1 and mp2    
                del merge_sets[0]
                del merge_sets[-1]

                # Create new intermediates for the rest
                for merge_set in merge_sets:
                    intermediate_mp = {}

                    for i, window in enumerate(merge_set):
                        edge_id = overlap[i]
                        intermediate_mp[edge_id] = TSSMergeItem(window, varying_indices[i], reverse_rungs[i])

                    interpolated_merge_proposals.append(intermediate_mp)

                # we only support point and line joins so there won't be
                # another mp with an overlap, so stop looking
                break

            #--------end of part to change for n-dimensional merges----
        return merge_proposals + interpolated_merge_proposals


    def _perform_window_merges(self, proposals):
        '''Given merge proposals, perform the merges.'''

        # 1. Construct new windows containing all the partial windows
        merged_windows = [TSSMultiEdgeWindow(mp) for mp in proposals]

        #print proposals

        # 2. Create nodes joining all border rungs so extended neighbors can be detected
        for mp in proposals:
            edges = list(mp.keys())

            # Get each set of rungs that needs to be merged
            pws_merger_rungs = [get_merger_rungs(mi) for mi in mp.values()]
            merger_rung_groups = itertools.zip_longest(*pws_merger_rungs, fillvalue=None)

            for mrg in merger_rung_groups:
                merge_pairs = [TSSNodeLocation(edge, tuple(mrg[i])) for i, edge in enumerate(edges) if mrg[i] != None]

                new_name = 'stitcher_node_edge_rung_pairs_{}'.format(merge_pairs)
                n = TSSNode(new_name)

                first_edge, first_rung_loc = merge_pairs[0].edge_id, merge_pairs[0].location
                first_rung_params = self.rungs[first_edge][first_rung_loc].values

                for nl in merge_pairs:
                    # If this partial window has a rung_loc in this group, add it to the node
                    # (if not, this merge proposal involves both a point and a line merge)
                    n.rung_set[nl.edge_id] = nl.location

                    rung_params = self.rungs[nl.edge_id][nl.location].values 

                    # Check that all rung parameters are the same for all rungs in this node
                    # This is the major merge-sanity assertion
                    assert param_equiv(rung_params, first_rung_params), "Parameter mismatch in rung merger for node {}, edge {} rung {} params {} != edge {} rung {} params {}".format(new_name, first_edge, first_rung_loc, "\n"+Ark(first_rung_params).format(), nl.edge_id, nl.location, "\n"+Ark(rung_params).format())

                self.nodes[new_name] = n

        return merged_windows

    def _clean_partial_windows(self, merge_proposals):
        '''Find all the partial windows referenced in the list of merge proposals and remove
            them from self.partial_windows. Use after merges have been processed to avoid
            both a partial window and a merged window being output for the same sets of rungs.'''

        for mp in merge_proposals:
            for edge_id, merge_item in mp.items():
                self.partial_windows[edge_id].remove(merge_item.partial_window)

    def build(self, ark_graph=None, filename=None):
        '''Generate the TSS graph. If ark_graph is supplied, all its
            edges and schedules are added. If ark_graph is None, the user
            should have already added all edges and schedules using the API.'''
        ark_graph = copy.deepcopy(ark_graph)

        if ark_graph:
            # 0. Construct edges
            # Not explicitly written to file, but needed as an intermediate step
            ark_edges = ark_graph['edges']
            
            # 1. Loop through each edge in ark_edges and add it
            for edge in ark_edges:
                edge_id = self.add_edge_ark(edge)

                for entry in edge['schedule']:
                    self.add_schedule_ark(edge_id, entry)

        # 2. Do all the merge steps. 
        self._propagate_fixed_parameters(self.nodes)
        proposals = self._propose_node_merges()
        extended_proposals = self._extend_merges(proposals, self.partial_windows)
        self.merged_windows = self._perform_window_merges(extended_proposals)
        self._clean_partial_windows(extended_proposals)
        self._insert_rung_neighbors()

        # Store the input graph
        self.input_ark = ark_graph if ark_graph else None

        # At this point the graph is fully read. 
        # 3. Serialize out.
        return self.cerealize(filename)
