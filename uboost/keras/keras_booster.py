import sys

from ..core import GradientBooster


def fine_tune_keras_ensemble(booster, epochs=1, verbose=0):
    ensemble_model = None
    if "tensorflow" in sys.modules:
        import tensorflow.keras as k
        import tensorflow.keras.backend as kb

        if any([isinstance(l, k.Model) for l in booster.learners]):
            X = k.Input(shape=booster.input_dim)
            BASE_SCORE = k.Input(shape=booster.output_dim)

            base_score = booster.base_score

            predictions = []
            for key in booster.learners.keys():
                model = booster.learners[key]
                if isinstance(model, k.Model):
                    predictions += [kb.expand_dims(booster.epsilons[key] * model(X), axis=-1)]
                else:
                    base_score += booster.epsilons[key] * model.predict(booster.x).reshape((-1,) + booster.output_dim)
                    pass
                pass
            predictions = kb.concatenate(predictions, axis=-1)
            ensemble_prediction = BASE_SCORE + kb.sum(predictions, axis=-1, keepdims=False)

            ensemble_model = k.Model([X, BASE_SCORE], ensemble_prediction)
            ensemble_model.compile(loss=booster.objective, optimizer=k.optimizers.Adam(1e-3, clipnorm=1.0))
            ensemble_model.fit([booster.x, base_score], booster.y, epochs=epochs, verbose=verbose)
            pass
        pass
    return ensemble_model


class KerasGradientBooster(GradientBooster):

    def _fit_learner(self):
        super(KerasGradientBooster, self)._fit_learner()
        fine_tune_keras_ensemble(self, epochs=1, verbose=1)
        pass

    pass
