## Описание проекта

Цель проекта - реализовать классические алгоритмы машинного обучения для укрепления понимания их работы.

В качестве бенчмарка использовались метрики качества алгоритмов sklearn. Если кастомный алгоритм сопоставим по качеству с алгоримтмом sklearn, то кастомная реализация считается успешной.

> **Note** \
> Все алгоритмы реализованы в упрощенной форме. Вопросы быстродействия оставлены за периметром проекта.

## Результаты

Были реализованы следующие алгоритмы:

- Linear regression
- Logistic regression
- Linear SVC (L2 hinge loss; SGD optimization)
- KNN Regressor/Classifier
- Random Forest Regressor/Classifier
- Gradient Boosting Machine Regressor/Classifier

Сравнение с бенчмарком:

`Регрессия`:

| Model                     | MSE           | R2       |
|---------------------------|---------------|----------|
| Custom OLS                 | 2462.036191   | 0.824928 |
| Sklearn OLS                | 2462.036489   | 0.824928 |
| Custom Lasso               | 2459.439009   | 0.825113 |
| Sklearn Lasso              | 2459.436680   | 0.825113 |
| Custom Ridge               | 2461.521346   | 0.824965 |
| Sklearn Ridge              | 2461.526869   | 0.824964 |
| Custom KNN uniform         | 4250.647848   | 0.697742 |
| Sklearn KNN uniform        | 4250.647848   | 0.697742 |
| Custom KNN distance        | 3868.797334   | 0.724895 |
| Sklearn KNN distance       | 3909.937129   | 0.721970 |
| Custom RF                  | 2870.074095   | 0.795913 |
| Sklearn RF                 | 2897.756595   | 0.793944 |
| Custom GBM                 | 2584.427222   | 0.816225 |
| Sklearn GBM                | 2584.550416   | 0.816216 |

`Классификация`:

| Model                     | Accuracy   | ROC AUC   |
|---------------------------|------------|-----------|
| Custom LogReg             | 0.732333   | 0.813961  |
| Sklearn LogReg            | 0.732667   | 0.813962  |
| Custom SVC                | 0.706000   | -         |
| Sklearn SVC               | 0.698667   | -         |
| Custom KNN uniform        | 0.789333   | 0.836363  |
| Sklearn KNN uniform       | 0.789333   | 0.836363  |
| Custom KNN distance       | 0.813667   | 0.868862  |
| Sklearn KNN distance      | 0.812000   | 0.868316  |
| Custom RF                 | 0.835667   | 0.879501  |
| Sklearn RF                | 0.835667   | 0.879591  |
| Custom GBM                | 0.826667   | 0.871994  |
| Sklearn GBM               | 0.827000   | 0.871815  |
