
import pytest
import torch


class PatchPooling(torch.nn.Module):
    def forward(self, batch: torch.Tensor, patch_lengths: torch.Tensor) -> torch.Tensor:
        B, S, D = batch.shape
        B_1, P = patch_lengths.shape

        assert B == B_1

        end_indexes = torch.cumsum(patch_lengths, dim=1).unsqueeze(2)
        start_indexes = end_indexes - patch_lengths.unsqueeze(2)

        token_positions = torch.arange(S, device=batch.device).unsqueeze(0).unsqueeze(0)

        mask = (token_positions >= start_indexes) & (token_positions < end_indexes)
        mask = mask.float()

        patch_sums = torch.bmm(mask, batch)

        lenghts = patch_lengths.unsqueeze(2)
        denominators = torch.where((lenghts > 0), lenghts, torch.ones_like(lenghts))

        means = patch_sums / denominators

        outoput = torch.where((lenghts > 0), means, -torch.ones_like(lenghts))

        return outoput







class TestPatchPooling:
    @pytest.mark.parametrize(
        "batch,patch_lengths,expected_output",
        [
            (
                torch.tensor(
                    [
                        [
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0],
                            [3.0, 3.0, 3.0, 3.0, 3.0],
                            [3.0, 3.0, 3.0, 3.0, 3.0],
                        ],
                        [
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [5.0, 5.0, 5.0, 5.0, 5.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                        ],
                        [
                            [6.0, 6.0, 6.0, 6.0, 6.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                        ],
                    ]
                ),
                torch.tensor([[3, 1, 2], [4, 1, 0], [1, 0, 0]]),
                torch.tensor(
                    [
                        [
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0],
                            [3.0, 3.0, 3.0, 3.0, 3.0],
                        ],
                        [
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [5.0, 5.0, 5.0, 5.0, 5.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                        ],
                        [
                            [6.0, 6.0, 6.0, 6.0, 6.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                        ],
                    ]
                ),
            )
        ],
    )
    def test_forward(
        self,
        batch: torch.Tensor,
        patch_lengths: torch.Tensor,
        expected_output: torch.Tensor,
    ) -> None:
        # given
        patch_pooling = PatchPooling()

        # when
        output = patch_pooling(batch=batch, patch_lengths=patch_lengths)

        # then
        assert torch.all(torch.isclose(output, expected_output))
