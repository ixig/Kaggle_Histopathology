"""kaggle_histopathology dataset."""

import tensorflow_datasets as tfds
from pathlib import Path
import csv
import tempfile
from PIL import Image


_DESCRIPTION = """\
Dataset from [Kaggle Histopathologic Cancer Detection]
(https://www.kaggle.com/competitions/histopathologic-cancer-detection/)

The data for this competition is a slightly modified version of the PatchCamelyon (PCam) 
benchmark dataset (the original PCam dataset contains duplicate images due to its probabilistic 
sampling, however, the version presented on Kaggle does not contain duplicates).

https://www.kaggle.com/competitions/histopathologic-cancer-detection/data
"""

_CITATION = """\
@ARTICLE{Veeling2018-qh,
  title         = "Rotation Equivariant {CNNs} for Digital Pathology",
  author        = "Veeling, Bastiaan S and Linmans, Jasper and Winkens, Jim and
                   Cohen, Taco and Welling, Max",
  month         =  jun,
  year          =  2018,
  archivePrefix = "arXiv",
  primaryClass  = "cs.CV",
  eprint        = "1806.03962"
}
"""

DATA_DIR = '../data'
CLASSES = ['NORMAL', 'TUMOR']

class KaggleHistopathology(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for kaggle_histopathology dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(None, None, 3), encoding_format='jpeg'),
            'label': tfds.features.ClassLabel(names=CLASSES),
            'id': tfds.features.Text()
        }),
        supervised_keys=('image', 'label'),
        homepage='https://www.kaggle.com/competitions/histopathologic-cancer-detection/data',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = Path(DATA_DIR)

    with open(path / "train_labels.csv", newline="") as f:
      reader = csv.reader(f)
      next(reader)
      labels = {fname: int(label) for fname, label in reader}
    
    return {
      'train': self._generate_examples(path / 'train', labels),
      # 'test': self._generate_examples(path / 'test', labels),
    }

  def _generate_examples(self, path, labels):
    """Yields examples."""
    for f in path.glob('*.tif'):
      im = Image.open(f)
      f_jpg = tempfile.NamedTemporaryFile()
      im.save(f_jpg, 'jpeg', quality=95)
      yield f.stem, {
          'image': Path(f_jpg.name),
          'label': CLASSES[labels[f.stem]],
          'id': f.stem
      }
      f_jpg.close()
