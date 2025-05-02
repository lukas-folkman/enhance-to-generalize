import os
import sys
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import config as C
from cv_tools import read_json, read_imgs_and_resize


def main():

    enhancement_dir_name = sys.argv[1]
    resize = int(sys.argv[2])

    reducer = PCA(n_components=10, random_state=0)

    for dataset_name in ['DeepFish', 'Jellytoring', 'GLOW_low_visibility_estuaries', 'S-UODAC']:
        print(f'\n{dataset_name}')
        annot_fn, img_dir = C.DATASETS[dataset_name]
        if enhancement_dir_name is not None and enhancement_dir_name.lower() != 'none':
            img_dir = os.path.join(os.path.dirname(img_dir), enhancement_dir_name)
        assert os.path.exists(img_dir)

        dataset = read_json(annot_fn, verbose=False)
        print(f'{dataset_name}: Reading and resizing to {resize} x {resize} pixels')
        img_arr, metadata = read_imgs_and_resize(dataset=dataset, img_dir=img_dir, resize=resize, group_splitter="_")
        print(f'{dataset_name}: Standard scaling before PCA')
        img_arr = StandardScaler().fit_transform(img_arr)
        print(f'{dataset_name}: Reducing with PCA')
        img_arr = reducer.fit_transform(img_arr)

        store_kwargs = {
            f'img_arr': img_arr,
            f'metadata': metadata
        }

        for n in [2, 3, 5, 10]:
            print(f'{dataset_name}: Calculating silhouette scores with {img_arr[:, :n].shape}')
            s = silhouette_score(img_arr[:, :n], labels=metadata[:, -1])
            store_kwargs[f'S{n}'] = s

        fn = os.path.join(C.RESULTS_DIR, f'{dataset_name}.PCA.{enhancement_dir_name}.{resize}px.npz')
        print(f'{dataset_name}: Saving {fn}')
        np.savez_compressed(fn, **store_kwargs)
        print(f'{dataset_name}: Done.')


if __name__ == '__main__':
    main()
