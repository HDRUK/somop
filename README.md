# somop

synthetic omop generator

## Usage

Use the config files to generate synthetic datasets

```
somop --config configs/ckd_antibodies.yaml
somop --config configs/conditions.yaml
somop --config configs/symptoms.yaml
somop --config configs/more_symptoms.yaml
```

## Load data into postgres

```
docker compose -f synthetic-omop.yaml
```
