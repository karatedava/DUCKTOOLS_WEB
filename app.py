from flask import Flask, render_template, request, redirect, url_for, flash
from pathlib import Path
import json

from src.growth_modeler.growth_modeler import GrowthModeler
from src.yield_predictor.yield_predictor import YieldPredictor
import src.utils as utils
from src.config import CONFIG_FOLDER, OUTPUT_DIR

app = Flask(__name__)
app.secret_key = "your-secret-key"  # Required for flashing messages

@app.route('/', methods=['GET', 'POST'])
def simulate():
    if request.method == 'POST':
        try:
            # Extract form data
            simulator = request.form.get('simulator').upper()
            config_file = request.form.get('config_file')
            generate_report = request.form.get('generate_report') == 'on'
            output_dir = request.form.get('output_dir', str(OUTPUT_DIR))

            # Validate inputs
            if simulator not in ['GM', 'YP']:
                flash(f"Invalid simulator: {simulator}. Must be 'GM' or 'YP'.", 'error')
                return redirect(url_for('simulate'))

            config_path = Path(CONFIG_FOLDER) / config_file
            if not config_path.is_file():
                flash(f"Config file {config_file} does not exist.", 'error')
                return redirect(url_for('simulate'))

            output_path = Path(output_dir)
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                flash(f"Cannot write to output directory {output_dir}: {e}", 'error')
                return redirect(url_for('simulate'))

            # Initialize and run model
            if simulator == 'GM':
                model = GrowthModeler(config_path, output_path)
            else:
                model = YieldPredictor(config_path, output_path)

            simulation_report_fname = model.run_simulation()

            # Generate report if requested
            report_files = None
            if generate_report and simulation_report_fname:
                # Ensure simulation_report_fname is a full path
                report_path = Path(simulation_report_fname)
                if not report_path.is_absolute():
                    report_path = output_path / simulation_report_fname
                
                report_files = utils.gen_report(report_path)

            return render_template('results.html', 
                                simulator=simulator, 
                                config_file=config_file, 
                                output_dir=output_dir, 
                                report_files=report_files)

        except Exception as e:
            flash(f"An error occurred: {str(e)}", 'error')
            return redirect(url_for('simulate'))

    # GET request: Render the simulation form
    config_files = [f.name for f in Path(CONFIG_FOLDER).glob('*.json')]
    return render_template('simulation_form.html', config_files=config_files,OUTPUT_DIR=str(OUTPUT_DIR))

if __name__ == '__main__':
    app.run(debug=False)