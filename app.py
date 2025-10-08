from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from pathlib import Path
import json
import numpy as np

from src.growth_modeler.growth_modeler import GrowthModeler
from src.yield_predictor.yield_predictor import YieldPredictor
import src.utils as utils
from src.config import CONFIG_FOLDER, OUTPUT_DIR

app = Flask(__name__)
app.secret_key = "your-secret-key"  # Required for flashing messages

@app.route('/')
def intro():
    return render_template('intro.html')

@app.route('/simulate_1', methods=['GET', 'POST'])
def simulate_1():
    if request.method == 'POST':
        try:
            # Extract form data
            simulator = request.form.get('simulator').upper()
            config_file = request.form.get('config_file')
            generate_report = request.form.get('generate_report') == 'on'
            output_dir = str(OUTPUT_DIR)

            cultivation_time = int(request.form['c_time'])
            limiting_biomass = int(request.form['lb'])
            hp_min = float(request.form['hp_min'])
            hp_max = float(request.form['hp_max'])
            hr_min = float(request.form['hr_min'])
            hr_max = float(request.form['hr_max'])
            w0_min = float(request.form['w0_min'])
            w0_max = float(request.form['w0_max'])

            # Define ranges using slider values
            HPs = np.arange(hp_min, hp_max + 1, 1)  # +1 to include max in range
            HRs = np.arange(hr_min, hr_max + 0.1, 0.1)  # +0.1 to include max
            W0s = np.arange(w0_min, w0_max + 100, 100)  # +100 to include max

            config_files = [f.name for f in Path(CONFIG_FOLDER).glob('*.json')]
            if hp_min > hp_max or hr_min > hr_max or w0_min > w0_max:
                flash('Minimum values must be less than or equal to maximum values.', 'error')
                return render_template('sim_form1.html', config_files=config_files, OUTPUT_DIR=str(OUTPUT_DIR))

            # Validate inputs
            if simulator not in ['GM', 'YP']:
                flash(f"Invalid simulator: {simulator}. Must be 'GM' or 'YP'.", 'error')
                return redirect(url_for('simulate_1'))

            config_path = Path(CONFIG_FOLDER) / config_file
            if not config_path.is_file():
                flash(f"Config file {config_file} does not exist.", 'error')
                return redirect(url_for('simulate_1'))

            output_path = Path(output_dir)
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                flash(f"Cannot write to output directory {output_dir}: {e}", 'error')
                return redirect(url_for('simulate_1'))

            # Initialize and run model
            if simulator == 'GM':
                model = GrowthModeler(cultivation_time, limiting_biomass, config_path, output_path, HPs, HRs, W0s)
            else:
                model = YieldPredictor(config_path, output_path, HPs, HRs, W0s)

            simulation_report_fname = model.run_simulation()

            # Generate report if requested
            report_files = None
            if generate_report and simulation_report_fname:
                report_path = Path(simulation_report_fname)
                if not report_path.is_absolute():
                    report_path = output_path / simulation_report_fname
                
                report_files = utils.gen_report(report_path)

            return render_template('results.html', 
                                simulator=simulator, 
                                config_file=config_file, 
                                output_dir=output_dir, 
                                report_files=report_files,
                                bm_csv_file=f'tmp/outputs/{simulation_report_fname}.biomass_df.csv',
                                hv_csv_file=f'tmp/outputs/{simulation_report_fname}.harvest_df.csv',
                                gs_csv_file=f'tmp/outputs/{simulation_report_fname}')

        except Exception as e:
            flash(f"An error occurred: {str(e)}", 'error')
            return redirect(url_for('simulate_1'))

    # GET request: Render the simulation form
    config_files = [f.name for f in Path(CONFIG_FOLDER).glob('*.json')]
    return render_template('sim_form1.html', config_files=config_files, OUTPUT_DIR=str(OUTPUT_DIR))

@app.route('/simulate_2', methods=['GET', 'POST'])
def simulate_2():
        
    return render_template('sim_form2.html',OUTPUT_DIR=str(OUTPUT_DIR))


@app.route('/<path:filename>')
def download_file(filename):
    return send_from_directory('./', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False)