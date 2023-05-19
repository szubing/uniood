import os
import faiss
import copy
import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import recall_score, average_precision_score

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (tensor): the inputs that's used to call the model.
            outputs (tensor): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., Accuracy)
                * value: a dict of {metric name: score}, e.g.: {"Accuracy": 0.80}
        """
        pass


class UniDAEvaluator(DatasetEvaluator):
    def __init__(self, n_source_classes, norm=True) -> None:
        self.ground_truths = []
        self.predicts = []
        self.predicts_without_ood = []
        self.features = []
        self.iid_scores = []
        self.n_source_classes = n_source_classes
        self.norm = norm
    
    def process(self, inputs, outputs, outputs_witoud_ood=None, iid_scores=None, features=None):
        if inputs is not None:
            self.ground_truths.append(inputs.cpu().detach())
        if outputs is not None:
            self.predicts.append(outputs.cpu().detach())
        if outputs_witoud_ood is not None:
            self.predicts_without_ood.append(outputs_witoud_ood.cpu().detach())
        if features is not None:
            self.features.append(features.cpu().detach())
        if iid_scores is not None:
            self.iid_scores.append(iid_scores.cpu().detach())

    def evaluate(self):
        result = {'OA': None, 
                  'AA': None, 
                  'OOD': None, 
                  'NMI': None,
                  'H-score': None, 
                  'H3-score': None, 
                  'UCR': None,
                  'Recall-all': None,
                  'Closed Set Accuracy': None,
                  'OSR Accuracy': None}
        
        grds = torch.cat(self.ground_truths)
        prds = torch.cat(self.predicts)

        label_set = set(grds.tolist())
        private_label_set = label_set - set(range(self.n_source_classes))
        n_target_private = len(private_label_set)

        np_grds = copy.copy(grds.numpy())
        np_prds = copy.copy(prds.numpy())
        target_private_indexs = [True if lab in private_label_set else False for lab in grds.tolist()]
        np_grds[target_private_indexs] = self.n_source_classes
        recall_avg_auc = recall_score(np_grds, np_prds, labels=np.unique(np_grds), average=None)

        # obtain Ovearll accuracy
        acc = np.mean(np_grds==np_prds)
        result['OA'] = float(100.*acc)

        # obtain Average classes score (may include ood class)
        result['AA'] = float(100.*recall_avg_auc.mean())
        result['Recall-all'] = list(100.*recall_avg_auc)

        # obtain H-score
        if n_target_private != 0:
            acc_shared = recall_avg_auc[:-1].mean()
            acc_ood = recall_avg_auc[-1]
            h_score = 2 * acc_ood * acc_shared / (acc_ood + acc_shared + 1e-5)
            result['OOD'] = 100.*acc_ood
            result['H-score'] = float(100.*h_score)

        # obtain NMI score
        if len(self.features) > 0:
            features = torch.cat(self.features)
            if self.norm:
                import torch.nn.functional as F
                features = F.normalize(features)
            if n_target_private != 0:
                private_pred, _ = run_kmeans(features[target_private_indexs], n_target_private, init_centroids=None, gpu=True)
                nmi = normalized_mutual_info_score(grds[target_private_indexs].numpy(), private_pred)
                result['NMI'] = float(100.*nmi)

        # obtain H3-score
        if n_target_private != 0 and len(self.features) > 0:
            h3_score = 3*(1/(1/(acc_shared+1e-5) + 1/(acc_ood+1e-5) + 1/(nmi+1e-5)))
            result['H3-score'] = float(100.*h3_score)

        # obtain closed set accuracy withou ood detection
        if len(self.predicts_without_ood) > 0:
            prds_without_ood = torch.cat(self.predicts_without_ood)
            np_prds_without_ood = copy.copy(prds_without_ood.numpy())
            shared_indexs = [False if id else True for id in target_private_indexs]
            if sum(shared_indexs) > 0:
                acc_without_ood = np.mean(np_grds[shared_indexs]==np_prds_without_ood[shared_indexs])
                recall_avg_auc_without_ood = recall_score(np_grds[shared_indexs], np_prds_without_ood[shared_indexs], labels=np.unique(np_grds[shared_indexs]), average=None)

                result_without_ood = {'OA': float(100.*acc_without_ood),
                                      'AA': float(100.*recall_avg_auc_without_ood.mean()),
                                      'Recall-all': list(100.*recall_avg_auc_without_ood)}
                result['Closed Set Accuracy'] = result_without_ood

                # obtain OOD detection/Openset recognition (OSR) accuracy
                if len(self.iid_scores) > 0 and sum(target_private_indexs) > 0:
                    iid_scores = torch.cat(self.iid_scores).numpy()

                    # Out-of-Distribution detction evaluation
                    x1, x2 = iid_scores[shared_indexs], iid_scores[target_private_indexs]
                    result_osr = metric_ood(x1, x2)['Bas']
                    
                    # OSCR
                    oscr_socre = compute_oscr(x1, x2, np_prds_without_ood[shared_indexs], np_grds[shared_indexs])

                    # Average precision
                    ap_score = average_precision_score([0] * len(x1) + [1] * len(x2), list(-x1) + list(-x2))

                    result_osr['OSCR'] = 100.*oscr_socre
                    result_osr['AUPR'] = 100.*ap_score

                    result['OSR Accuracy'] = result_osr
        
        # obtain UCR
        if  result['OSR Accuracy'] is not None:
            result['UCR'] = result['OSR Accuracy']['OSCR']
        elif result['Closed Set Accuracy'] is not None:
            result['UCR'] = result['Closed Set Accuracy']['OA']

        return result
    
    def reset(self):
        self.ground_truths = []
        self.predicts = []
        self.predicts_without_ood = []
        self.features = []
        self.iid_scores = []



#############################################################################################################
# Common functions to use
def run_kmeans(L2_feat, ncentroids, init_centroids=None, gpu=False, min_points_per_centroid=1):
    dim = L2_feat.shape[1]
    kmeans = faiss.Kmeans(d=dim, k=ncentroids, gpu=gpu, niter=20, verbose=False, \
                        nredo=5, min_points_per_centroid=min_points_per_centroid, spherical=True)
    if torch.is_tensor(L2_feat):
        L2_feat = L2_feat.cpu().detach().numpy()
    kmeans.train(L2_feat, init_centroids=init_centroids)
    _, pred_centroid = kmeans.index.search(L2_feat, 1)
    pred_centroid = np.squeeze(pred_centroid)
    return pred_centroid, kmeans.centroids


def get_curve_online(known, novel, stypes = ['Bas']):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known),np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l+1] = tp[stype][l]
                    fp[stype][l+1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1
                    fp[stype][l+1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95


def metric_ood(x1, x2, stypes = ['Bas'], verbose=False):
    tp, fp, tnr_at_tpr95 = get_curve_online(x1, x2, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()
        
        # TNR
        mtype = 'TNR'
        results[stype][mtype] = 100.*tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        results[stype][mtype] = 100.*(-np.trapz(1.-fpr, tpr))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = 100.*(.5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max())
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = 100.*(-np.trapz(pin[pin_ind], tpr[pin_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = 100.*(np.trapz(pout[pout_ind], 1.-fpr[pout_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
            print('')
    
    return results


def compute_oscr(x1, x2, pred_in, labels_in):
    m_x1 = np.zeros(len(x1))
    m_x1[pred_in == labels_in] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values
    
    CCR = [0 for x in range(n+2)]
    FPR = [0 for x in range(n+2)] 

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n-1):
        CC = s_k_target[k+1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n+1] = 1.0
    FPR[n+1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n+1):
        h =   ROC[j][0] - ROC[j+1][0]
        w =  (ROC[j][1] + ROC[j+1][1]) / 2.0

        OSCR = OSCR + h*w

    return OSCR