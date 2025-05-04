import torch
import numpy as np
from data_solver import generate_linear_system_batch
from data_generator import generate_quadratic_data, prepare_icl_data

# Generate test data for linear function
dim = 9
batch_size = 100
condition_number = 5

# Linear function test data
A_batch, x_batch, b_batch = generate_linear_system_batch(batch_size, dim, condition_number, mode='solver')
linear_test_data = {
    'dim_9': {
        'A_batch': A_batch,
        'x_batch': x_batch,
        'b_batch': b_batch
    }
}

# Save test data
torch.save(linear_test_data, 'results/data/test_data.pt') 