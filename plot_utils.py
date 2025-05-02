import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

import config as C
import cv_tools as utils


def pps_df_to_latex(df, datasets=C.MAIN_DATASETS, placeholder=None, rename=True, decorate_bold=True, decorate_ital=True):
    def r_to_str(row, d, decorate_bold=decorate_bold, decorate_ital=decorate_ital):
        mAP = f"{row[f'mAP_{d}'] * 100:.1f}"
        std = f"{row[f'std_{d}'] * 100:.1f}"
        rank = f"{row[f'rank_{d}']:.0f}"
        if decorate_bold or decorate_ital:
            decor = 'textbf' if row[f'rank_{d}'] == 1 and decorate_bold else 'textit' if row[f'rank_{d}'] == 2 and decorate_ital else None
            if decor:
                mAP = f"\\{decor}" + "{" + mAP + "}"
                std = f"\\{decor}" + "{" + std + "}"
                rank = f"\\{decor}" + "{" + rank + "}"

        return f"& ${mAP} \pm {std}$ & {rank}"

    if placeholder is None:
        placeholder = ''
    if rename and not isinstance(rename, dict):
        rename = C.RENAME_PPS
    table = []
    for pps, row in df.iterrows():
        result_str = " ".join([f'{r_to_str(row, d):<21}' for d in datasets])
        table.append(
            f"{rename.get(pps, pps):>14}{f' & {placeholder[pps]:>6}' if isinstance(placeholder, dict) else placeholder} {result_str}\\\\"
        )
    return table


def get_results(model, dataset, eval_type, augm, model_select, split, pickle_prefix, concat=None):
    results = utils.pickle_load(f'{pickle_prefix}.{model}.{dataset}.{eval_type}.{augm}{f".concat{concat}" if concat is not None else ""}.pckl')
    train_data = f'{dataset}.{"CV." if eval_type in ["Groups", "NoGroups"] else ""}{eval_type}{f".{augm}" if augm else ""}'
    assert results[train_data][model][model_select][split] is not None
    
    if dataset == "S-UODAC" and model == "yolo3":
        for seed in results[train_data][model][model_select][split]:
            if results[train_data][model][model_select][split][seed].gt_fn is None:
                results[train_data][model][model_select][split][seed].gt_fn = os.path.join(
                    '..', 'results',
                    'FINAL_S-UODAC', f'yolo3.{dataset}.NoValReproE100{f".{augm}" if augm else ""}.seed{seed}',
                    'test',
                    'annotations.fixed.test.json.gz'
                )
            if results[train_data][model][model_select][split][seed].dt_fn is None:
                results[train_data][model][model_select][split][seed].dt_fn = os.path.join(
                    '..', 'results',
                    'FINAL_S-UODAC', f'yolo3.{dataset}.NoValReproE100{f".{augm}" if augm else ""}.seed{seed}',
                    'test',
                    'last_predictions.fixed.json.gz'
                )
    
    return results[train_data][model][model_select][split]


def pps_pr_curves(pickle_prefix, augms, models, datasets, model_select, split, eval_type,
                  linestyles='auto', palettes=[C.DARK, C.LIGHT], highlight=None, highlight_palettes=None,
                  f1_values=None, ax=None):
    if isinstance(pickle_prefix, str):
        pickle_prefix = [pickle_prefix] * len(augms)
    else:
        assert utils.is_iterable(pickle_prefix) and len(pickle_prefix) == len(augms)
    if isinstance(eval_type, str):
        eval_type = [eval_type] * len(augms)
    else:
        assert utils.is_iterable(eval_type) and len(eval_type) == len(augms)
    if linestyles == 'auto':
        linestyles = ['--', '-'] * int(np.ceil(len(augms) / 2))
    elif utils.is_iterable(linestyles):
        assert len(linestyles) == len(augms)
    else:
        linestyles = [linestyles] * len(augms)
    if highlight_palettes is None:
        highlight_palettes = palettes

    results = {}
    for color, dataset in enumerate(
        datasets
    ):
        results[dataset] = {
            model: {
                f'{augm}.{evt}': get_results(model=model, dataset=dataset, eval_type=evt, augm=augm,
                                  model_select=model_select, split=split, pickle_prefix=pfx)
                for augm, evt, pfx in zip(augms, eval_type, pickle_prefix)
            } for model in models
        }

        for model, palette, highlight_palette in zip(
            models,
            palettes,
            highlight_palettes
        ):
            for augm, evt, ls in zip(
                augms,
                eval_type,
                linestyles
            ):
                res = results[dataset][model][f'{augm}.{evt}']
                # stack the different seeds and average across seeds
                precisions = np.stack([res[s].precision for s in res]).mean(axis=0)
                # precision is [R, K] (101 recall thresholds x K categories)
                assert precisions.ndim == 2 and precisions.shape[0] == 101
                ax = utils.plot_PR_curve(
                    precision=precisions,
                    recall=np.arange(0, 1.01, 0.01),
                    catIds=None, all_catIds=None, average_cats=True,
                    ls=ls, color=color,
                    palette=palette if highlight is None or augm != highlight else highlight_palette,
                    f1_curves=f1_values is not None, f1_values=f1_values,
                    f1_format="$F_1$={:0.1f}", f1_fontsize=C.SMALL_FONT,
                    label=f'{model:<5} '
                          f'{C.RENAME_DATASETS.get(dataset, dataset):<13} '
                          f'{C.RENAME_PPS.get(augm, augm):<13} '
                          f'mAP={precisions.mean():.2f}',
                    ax=ax
                )
                f1_values = None
                ax.set_xticks(np.arange(0, 1.1, 0.2))
                ax.legend(bbox_to_anchor=(1.15, 1), prop=dict(family='Courier'))
    sns.despine()
    return ax, results


def sigmas_pointplot(df, values_col, idx=None, show_average=False, legend_visible=True, ylim=None, ax=None):
    # df0 = df.groupby('pps')[values_col].mean().sort_values()
    # idx = df0.index
    # df0 = df0.to_frame().reset_index()
    # df0['model'] = 'average'
    # df1 = df.loc[df['model'] == 'yolo8'].set_index('pps').loc[idx].reset_index()
    # df2 = df.loc[df['model'] == 'dt2'].set_index('pps').loc[idx].reset_index()
    # df = pd.concat([df0, df1, df2], axis=0).reset_index(drop=True)
    df['model'] = df['model'].transform(lambda x: C.MODEL_NAMES[x] if x in C.MODEL_NAMES else x)
    # idx = np.argsort([(np.sum([int(x) for x in X[1:]]) + (250 if X[0].startswith('FK') else 0)) if len(X) > 2 else (int(X[1]) - 250) for X in df['pps'].str.split('_', expand=False)])
    # df = df.iloc[idx]

    df = df.set_index('pps')
    if idx is not None:
        df = df.loc[np.asarray(idx)]
    ax = sns.pointplot(data=df.reset_index(), x='pps', y=values_col, hue='model', errorbar='sd',
                       hue_order=[C.MODEL_NAMES['dt2'], C.MODEL_NAMES['yolo8']] + (['average'] if show_average else []),
                       palette=np.asarray(sns.color_palette('deep'))[[0, 1, 7]], ax=ax)
    return ax
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel('Mean average\nprecision (mAP$_{50}$)')
    # ax.set_xticklabels([f"[{', '.join(x.get_text().split('_')[1:])}]{' with full kernel' if x.get_text().startswith('FK-') else ''}" for x in ax.get_xticklabels()])
    # ax.set_xticklabels([x.get_text().replace(' with', '\nwith') for x in ax.get_xticklabels()])
    ax.set_xlabel('MSRCR $\sigma$ values')
    ax.legend(title='Model').set_visible(legend_visible)
    if ylim is not None:
        ax.set_ylim(ylim)
    # sns.despine()
    return ax


def black_boxplot_kwargs():
    return {
        'boxprops':{'facecolor':'none', 'edgecolor':'k'},
        'medianprops':{'color':'k'},
        'whiskerprops':{'color':'k'},
        'capprops':{'color':'k'}
    }
