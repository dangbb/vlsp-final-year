from __future__ import annotations

import os
import numpy as np
import pandas as pd

from rouge import Rouge


class RougeN:
    def __init__(self, p: float | np.float, r: float | np.float, f1: float | np.float):
        self.p: float | np.float = p
        self.r: float | np.float = r
        self.f1: float | np.float = f1

    def __str__(self):
        return "p:\t{}\nr:\t{}\nf1:\t:{}".format(
            self.p,
            self.r,
            self.f1)


class RougeScore:
    def __init__(
            self,
            rouge1: RougeN,
            rouge2: RougeN,
            rougeL: RougeN,
    ):
        self.rouge1: RougeN = rouge1
        self.rouge2: RougeN = rouge2
        self.rougeL: RougeN = rougeL

    def __str__(self):
        return "Rouge 1:\n{}\nRouge 2:\n{}\nRouge L:\n{}".format(self.rouge1, self.rouge2, self.rougeL)


class PipRouge:
    def __init__(self):
        self.rouge = Rouge()

    def __call__(self, predict: str, ref: str) -> RougeScore:
        result = self.rouge.get_scores(predict, ref)

        return RougeScore(
            RougeN(result[0]['rouge-1']['p'], result[0]['rouge-1']['r'], result[0]['rouge-1']['f']),
            RougeN(result[0]['rouge-2']['p'], result[0]['rouge-2']['r'], result[0]['rouge-2']['f']),
            RougeN(result[0]['rouge-l']['p'], result[0]['rouge-l']['r'], result[0]['rouge-l']['f']),
        )


class ScoreSummary:
    def __init__(self, name: str):
        self.name = name
        self.df = pd.DataFrame(columns=[
            'cluster_id',
            'rouge_1_p',
            'rouge_1_r',
            'rouge_1_f',
            'rouge_2_p',
            'rouge_2_r',
            'rouge_2_f',
            'rouge_l_p',
            'rouge_l_r',
            'rouge_l_f',
        ])

    def add_score(self, cluster_id: int, score: RougeScore):
        self.df = self.df.append({
            'cluster_id': cluster_id,
            'rouge_1_p': score.rouge1.p,
            'rouge_1_r': score.rouge1.r,
            'rouge_1_f': score.rouge1.f1,
            'rouge_2_p': score.rouge2.p,
            'rouge_2_r': score.rouge2.r,
            'rouge_2_f': score.rouge2.f1,
            'rouge_l_p': score.rougeL.p,
            'rouge_l_r': score.rougeL.r,
            'rouge_l_f': score.rougeL.f1,
        }, ignore_index=True)

    def save_report(self, path: str):
        print("Start to save report {} to: {}".format(
            self.name,
            path,
        ))
        try:
            if not os.path.exists(path):
                os.mkdir(path)

            self.df.to_csv(os.path.join(path, self.name + '.csv'), index=False)

            summary_df = pd.DataFrame(columns=[
                'name',
                'mean',
                'min',
                'max',
                'std',
            ])

            metric_cols = [
                'rouge_1_p',
                'rouge_1_r',
                'rouge_1_f',
                'rouge_2_p',
                'rouge_2_r',
                'rouge_2_f',
                'rouge_l_p',
                'rouge_l_r',
                'rouge_l_f', ]

            for col in metric_cols:
                describe = self.df[col].describe()
                summary_df = summary_df.append({
                    'name': col,
                    'mean': describe['mean'],
                    'min': describe['min'],
                    'max': describe['max'],
                    'std': describe['std'],
                }, ignore_index=True)
            summary_df.to_csv(
                os.path.join(path, self.name + '-summary.csv'),
                index=False
            )

            print("Save report complete")
        except Exception as e:
            print("Save report failed: ", e)


if __name__ == "__main__":
    evaluator = PipRouge()
    score = evaluator(
        """??? T??y Ban_Nha v?? B???_????o_Nha , g???n 750 v??? thi???t_m???ng li??n_quan ?????n n???ng n??ng ???? ???????c b??o_c??o trong giai_??o???n s??ng nhi???t c??n_qu??t .""",
        """Trong b???n b??o_c??o ?????c_bi???t : " S??? n??ng l??n to??n_c???u ng?????ng 1,5_????? C " , ???y_ban Li??n_ch??nh_ph??? v??? Bi???n_?????i Kh??_h???u nh???n_m???nh : " Ki???m_ch??? s??? n??ng l??n to??n_c???u ??? ng?????ng 1,5 thay_v?? 2 ????? C s??? l??m b???t ??i kho???ng 420 tri???u ng?????i ph???i th?????ng_xuy??n ch???u nh???ng ?????t s??ng nhi???t c???c_??oan v?? gi???m kho???ng 65 tri???u ng?????i ph???i ti???p_x??c v???i nh???ng ?????t s??ng nhi???t nguy_hi???m t???i t??nh_m???ng ."""
    )
    print(score)

    rouge_summary = ScoreSummary("testing")
    rouge_summary.add_score(1, score)
    rouge_summary.add_score(2, score)
    rouge_summary.add_score(3, score)
    rouge_summary.save_report("data/result")