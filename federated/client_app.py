from flwr.client import ClientApp
from flwr.common import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)

from task import get_model


app = ClientApp()


@app.train()
def train(message: Message, context: Context):

    # Get hospital/node information
    node_id = context.node_id

    print()
    print("=" * 40)
    print(f"Hospital Node {node_id}")
    print("=" * 40)

    # Create local model
    model = get_model()

    print("Received global 3D U-Net model.")
    print("Local model created.")

    # ------------------------------------------------
    # WEEK 1 ONLY
    # We are testing communication.
    # Actual local MRI training comes later.
    # ------------------------------------------------

    print("Local training simulation completed.")

    # Return the same model for now
    arrays = ArrayRecord(model.state_dict())

    metrics = MetricRecord({
        "hospital_id": str(node_id),
        "local_training": 1,
    })

    content = RecordDict({
        "arrays": arrays,
        "metrics": metrics,
    })

    return Message(
        content=content,
        reply_to=message
    )