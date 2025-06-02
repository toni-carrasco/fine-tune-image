from transformers import TrainerCallback

class StepEvalAndEarlyStopCallback(TrainerCallback):
    """
    Callback que:
      1) Cada eval_steps pasos marca should_evaluate = True para llamar a .evaluate().
      2) Tras cada evaluate, compara la pérdida de validación con la mejor encontrada.
         - Si mejora (lower loss), la guarda y reinicia el contador de paciencia.
         - Si no mejora, incrementa el contador. Cuando alcance `patience`, escribe should_training_stop=True.
    """
    def __init__(self, eval_steps: int, patience: int = 2):
        self.eval_steps = eval_steps
        self.patience = patience

        # Nombres internos para trackear
        self.best_loss = float("inf")
        self.num_bad_steps = 0  # cuántas evaluaciones consecutivas sin mejora

    def on_step_end(self, args, state, control, **kwargs):
        """
        Se llama justo después de cada paso de entrenamiento. Si global_step % eval_steps == 0,
        indicamos a Trainer que ejecute evaluate() en vez de continuar entrenando.
        """
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            control.should_evaluate = True
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Se llama justo después de Trainer.evaluate(). Aquí `metrics` contiene
        la pérdida (key "eval_loss") y otras métricas. Comparamos la pérdida ...
        """
        if metrics is None:
            return control

        # Nuevamente, el Trainer almacena la pérdida bajo "eval_loss"
        current_loss = metrics.get("eval_loss")

        if current_loss is None:
            # Si no hay "eval_loss", no hacemos nada
            return control

        # Compara con la mejor pérdida hasta ahora
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.num_bad_steps = 0
        else:
            self.num_bad_steps += 1

        # Si hemos superado la paciencia, solicitamos detener el entrenamiento
        if self.num_bad_steps >= self.patience:
            control.should_training_stop = True

        return control
