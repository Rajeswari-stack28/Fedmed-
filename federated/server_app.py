from flwr.server import ServerApp
from flwr.serverapp import Grid, Context

from flwr.app import ArrayRecord, ConfigRecord
from flwr.serverapp.strategy import FedAvg

from task import get_model


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):

    print()
    print("=" * 50)
    print("              FedMed Server")
    print("=" * 50)

    # Create the initial global model
    model = get_model()

    print("Global 3D U-Net created.")

    # Convert model to Flower ArrayRecord
    arrays = ArrayRecord(model.state_dict())

    # Read number of federated rounds
    num_rounds = context.run_config[
        "num-server-rounds"
    ]

    print(
        f"Federated rounds: {num_rounds}"
    )

    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=3,
        min_available_nodes=3,
    )

    print("Waiting for 3 hospital nodes...")

    # Start federated training
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord({}),
    )

    print()
    print("=" * 50)
    print("Federated run completed.")
    print("=" * 50)