import tss
import numpy as np

class Gaussian:
    '''A convenience class wrapping the gaussians model'''
    def __init__(self, graph_ark, params=None, directory=None, **kwargs):
        '''
Any additional named arguments passed in **kwargs will be passed through to
tss.Sampler as well.

Args:
    params (dict): A series of parameters to be passed through to tss.Sampler
    directory (str): The directory in which to write TSS ETR output
        '''
        if(params is None):
            params = {}
        for k,v in kwargs.items():
            if k not in params: 
                params[k] = v
        self.directory = directory
        self.sampler = tss.Sampler(graph_ark, **params)
        self.sampler.initGaussiansModel() 
    
    
    def __enter__(self):               
        import os
        if(isinstance(self.directory, str)):
            self._popdir = os.getcwd()
            os.makedirs(self.directory, exist_ok = True)
            os.chdir('{}'.format(self.directory))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import os
        self.sampler.stop()
        
        # if we cd'ed into the directory in __enter__, cd back out
        if(isinstance(self.directory, str)):
            os.chdir(self._popdir)
    
    def step(self, *args, **kwargs):
        '''
Pass arguments through to tss.Sampler.step
        '''
        self.sampler.step(*args, **kwargs)
    
    def trajectory(self, fields=None):
        '''
Extract all or a subset of TSS fields from the output ETR

Args:
    fields (list): A list of fields to store.  If None, all fields are stored.

Returns:
    list of dict: A dict mapping fields to keys for every frame in the ETR.
'''
        from msys.molfile import DtrReader as DR
        if(self.directory is None):
            data = DR('0/energy.etr')    
        else:
            data = DR('{}/0/energy.etr'.format(self.directory))
            
        curr_frame = {}
        traj = []
        
        for i in range(data.nframes):
            data.frame(i, keyvals=curr_frame)
            if(fields is None):
                fields = curr_frame.keys()
            traj.append({key:curr_frame[key] for key in fields})
        return traj

    def launch(self):
        '''
Get simulation context manager

Returns:
    context manager: Changes the current working directory to the trajectory
        output directory for the duration of the scope.
'''
        return self
        
def delta_free_energies(trajectory):
    '''
Extract the end-to-end delta free energies from a trajectory returned from a
Gaussian.trajectory call.

Args:
    trajectory (list of dict): Time series of dicts returned from Gaussian
        trajectory call, including at least the TSS_FE field

Returns:
    numpy array: dG between first and last rungs for each time.
    '''

    assert(isinstance(trajectory[0], dict)), "Trajectory must be a list of dicts with keys being the TSS keys."
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        dg = [frame['TSS_FE'][-1] - frame['TSS_FE'][0] for frame in trajectory]
    return np.array(dg)
