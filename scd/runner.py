import argparse
import h5py
import os
import pickle as pkl
import torch
import typer
import numpy as np
from pathlib import Path
from scd.config.structures import set_random_seed, Config
from scd.models.scd import SwarmContrastiveDecomposition
from motor_unit_toolbox import utils, props, spike_comp
from typing import Union, Optional

app = typer.Typer()

def load_sim_emg(file_save):
    data = dict.fromkeys(['emg','spikes','spikes_muscles','rms', 'noise','fs','angle_profile','force_profile','timestamps','ch_map'])
    with h5py.File(file_save, 'r') as h5:
        for key in data.keys():
            if key == 'spikes_muscles':
                data[key] = [n.decode("ascii", "ignore") for n in h5[key][()]]
            else:
                data[key] = h5[key][()]

    return data

@app.command()
def run_scd_sim(
    file_path: Path,
    save_path: Optional[Path] = None,
    *,
    square_sources: bool = True,
    max_iterations: int = 50,
    ext_fact: int = 5,
):  
    # Load data
    sim_data = load_sim_emg(file_path)
    
    # Pass EMG data [samples, chs]
    neural_data = torch.Tensor(sim_data['emg'])

    # Set config
    set_random_seed(seed=42)
    config = Config(
        square_sources=square_sources,
        max_iterations=max_iterations,
        extension_factor=ext_fact,
        sampling_frequency=2048,
        device='cuda',
        ica_patience=100,
        output_source_plot=False,
        output_final_source_plot=False,
    )

    # Select data
    if config.end_time == -1:
        neural_data = neural_data[config.start_time * config.sampling_frequency : , :]
    else:
        neural_data = neural_data[config.start_time * config.sampling_frequency : config.end_time * config.sampling_frequency, :]

    # Initiate the model and run
    model = SwarmContrastiveDecomposition()
    predicted_timestamps, dictionary = model.run(neural_data, config)

    # Get unique sources
    good_units = [i for i, x in enumerate(dictionary['source_type']) if x == 'good']

    # Format data
    spikes = utils.firings_to_binary(dictionary['timestamps'], neural_data.shape[0])[:, good_units]
    sources = np.concatenate(dictionary['source'],axis=1)[:, good_units]

    # Compute properties
    t = sim_data['timestamps']
    dictionary['dr_good'] = props.get_discharge_rate(spikes, t)
    dictionary['cov_good'] = props.get_coefficient_of_variation(spikes, t)
    dictionary['sil_good'] = props.get_silhouette_measure(spikes, sources, ext_fact=ext_fact)
    dictionary['pnr_good'] = props.get_pulse_to_noise_ratio(spikes, sources, ext_fact=ext_fact)

    # RoA with simulated sources
    dictionary['roa_good'], dictionary['roa_pair_idx'], dictionary['roa_pair_lag'] = spike_comp.rate_of_agreement(
        spikes, sim_data['spikes'], tol_spike_ms=0.5, tol_train_ms=40
    )

    # Save dictionary
    if save_path is not None:
        with open(save_path, "wb") as f:
            pkl.dump(dictionary, f)

    return dictionary

if __name__ == "__main__":
    app()