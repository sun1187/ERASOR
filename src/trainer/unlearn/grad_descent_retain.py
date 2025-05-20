from trainer.unlearn.base import UnlearnTrainer


class GradDescent_retain(UnlearnTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        outputs = model(**retain_inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
