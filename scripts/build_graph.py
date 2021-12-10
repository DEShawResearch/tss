#!/usr/bin/env python

import os
import sys
import argparse
from ark import Ark, File
from tss.tss_graph_builder import __doc__ as description
from tss import GraphBuilder

def main():
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('ark', help='The input ark file to process')
    parser.add_argument('-o', '--output', help='The output filename for the generated graph (default is [input].out)')
    parser.add_argument('-w', '--write-input-copy', help='Output filename for a copy of the input ark containing a pointer to the graph ark')

    args = parser.parse_args()
    execute(parser, args.ark, args.output, args.write_input_copy)


def execute(parser, ark, output, write_input_copy):
    arkl = Ark.load(ark)

    if not output:
        output = "{}.out".format(ark)

    ark_key = 'integrator.times_square'
    if ark_key in arkl:
        gr = GraphBuilder(arkl[ark_key])
    else:
        parser.error(f"Input ark must contain {ark_key}")

    # write out graph ark
    gr.build(filename = output)

    # write out copy of the input with reference to the new graph ark
    graph_key = '{}.graph.file'.format(ark_key)

    if write_input_copy:
        if os.path.isabs(output):
            output_path = output
        else:
            output_path = os.path.relpath(output, os.path.dirname(write_input_copy))

        # adjust ark keys in the output file
        arkl[graph_key] = output_path
        arkl.pop('{}.edges'.format(ark_key), None)
        arkl.pop('{}.blocks'.format(ark_key), None)

        arkl.save(write_input_copy, open_tables=True)


if __name__ == '__main__':
    main()
