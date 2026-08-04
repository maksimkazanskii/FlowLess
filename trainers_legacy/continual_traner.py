class ContinualTrainer:

    def __init__(
            self,
            model,
            dataset,
            method,
            analyzers
    ):

        self.model = model
        self.dataset = dataset
        self.method = method
        self.analyzers = analyzers

    def run(self):

        for task_id in range(
                self.dataset.num_tasks()
        ):

            trainset = self.dataset.get_task(
                task_id
            )

            self.method.before_task(
                self.model
            )

            self.train_task(trainset)

            self.method.after_task(
                self.model
            )

            self.collect_geometry(
                task_id
            )