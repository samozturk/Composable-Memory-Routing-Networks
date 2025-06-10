# file: tests/test_model.py
import torch
import unittest
from cmrn import CMRN, CMRNConfig

class TestCMRN(unittest.TestCase):
    def test_forward_pass(self):
        """Test a single forward pass with the correct output shapes."""
        config = CMRNConfig(
            vocab_size=1000,
            d_model=128,
            num_experts=4,
            expert_types=['general', 'symbolic'],
            memory_size=10000,
            memory_dim=128,
            routing_k=2
        )
        model = CMRN(config)

        # Create a dummy input tensor
        input_ids = torch.randint(0, config.vocab_size, (2, 32)) # (B, L)

        # Perform forward pass
        outputs = model(input_ids)

        # Check output shapes
        self.assertEqual(outputs.logits.shape, (2, 32, config.vocab_size))
        self.assertEqual(outputs.routing_weights.shape, (2, 32, config.num_experts))
        
        # Check that routing weights for each token sum to 1
        self.assertTrue(torch.allclose(
            outputs.routing_weights.sum(dim=-1), torch.ones(2, 32)
        ))

        print("Forward pass test successful!")

if __name__ == '__main__':
    unittest.main()