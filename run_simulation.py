from pathlib import Path
import argparse

from src.growth_modeler.growth_modeler import GrowthModeler
from src.yield_predictor.yield_predictor import YieldPredictor
import src.utils as utils

from src.config import *

def main(args):

    # Validate simulator
    simulator = args.simulator.upper()  # Normalize to uppercase
    if simulator not in ['GM', 'YP']:
        raise ValueError(f"Invalid simulator: {simulator}. Must be 'GM' or 'YP'.")
    
    # Validate config file
    config_file = Path(CONFIG_FOLDER / args.config)
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file {config_file} does not exist.")

    # Validate output directory
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    except PermissionError as e:
        raise PermissionError(f"Cannot write to output directory {output_dir}: {e}")
    
    # Initialize model based on simulator
    if simulator == 'GM':
        model = GrowthModeler(config_file, output_dir)
    else:  # simulator == 'YP'
        model = YieldPredictor(config_file, output_dir)
    
    simulation_report_fname = model.run_simulation()
    if args.report and simulation_report_fname:
        utils.gen_report(simulation_report_fname)

def _parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--simulator',
        type=str,
        choices=['GM', 'YP'],
        default=DEFAULT_SIMULATOR,
        help="Simulator to use: 'GM' (Growth Modeler) or 'YP' (Yield Predictor). Default: GM"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(DEFAULT_CONFIG_FILE),
        help='Path to configuration file'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        default=True,
        help='generate a graphical report (default: True)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(OUTPUT_DIR),
        help='directory for outputs'
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = _parse_arguments()
    main(args)