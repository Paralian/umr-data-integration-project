## Team

Team Name: TODO

Members:
 - Juan Fernando Maya
 - Ilya Zykov

## Getting started:
```bash
git init
git clone https://github.com/Paralian/umr-data-integration-project-the-TODO-team.git
```

## Structure

This data integration project has four components: preparation, data
integration, data cleaning, and showcase.

- `0_datasets`: We recommend to read and write source, intermediate, and final datasets from `0_datasets` directory. Its content is gitignored (i.e., it will not be tracked by Git), use [Hessenbox](https://hessenbox.uni-marburg.de) to sync-and-share your datasets.
- `1_preparation`: Code to extract and load your datasets into a common space
  (e.g., sqlite or Jupyter notebook). You might also want to write code to explore the
datasets.
- `2_integration`: Experiments to integrate your datasets and a program that
  implements your final integration pipeline.
- `3_cleaning`: Experiments to clean your integrated dataset and a program that
  implements your final data cleaning pipeline.
- `3_showcase`: Code to implement your showcase. You can already run it against
  the results of your data integration pipeline, and, finally, against your
integrated and cleaned data.

## Documentation

All code documentation and instructions should be placed in this `README.md`;
feel free to erase this intro text.
