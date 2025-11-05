"""Example manual run for the multi-agent pipeline.

Usage (ensure dependencies installed; heavy models will download on first use):
python -m backend.agents.example_run --file ./data/sample.pdf --claim-amount 2500
"""
from __future__ import annotations
import argparse
from .pipeline import FraudClaimPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Path to claim document (pdf/image)')
    parser.add_argument('--claim-amount', type=float, default=2500.0)
    parser.add_argument('--policy-duration', type=int, default=12)
    parser.add_argument('--previous-claims', type=int, default=0)
    args = parser.parse_args()

    pipeline = FraudClaimPipeline()
    meta = {
        'claim_amount': args.claim_amount,
        'policy_duration': args.policy_duration,
        'previous_claims': args.previous_claims,
    }
    result = pipeline.run(args.file, meta)
    import json, pprint
    pprint.pp(result)

if __name__ == '__main__':
    main()
