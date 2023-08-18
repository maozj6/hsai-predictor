
import numpy as np
from netcal.scaling import TemperatureScaling
from netcal.metrics import ECE
from netcal.binning import HistogramBinning, IsotonicRegression, ENIR, BBQ
from netcal.scaling import LogisticCalibration, TemperatureScaling, BetaCalibration
from netcal.metrics import ACE, ECE, MCE
from netcal.presentation import ReliabilityDiagram
import matplotlib.pyplot as plt


def main(step,testpath,validpath):
    n_bins = 10
    bins = 10
    hist_bins = 20

    ece = ECE(n_bins)
    data_vali = np.load(testpath)
    data = np.load(validpath)
    totalsft=data["lbl"]
    build_set_sm = data["sft"][step]
    build_set_gt = data['lbl'][step]
    sft_vali = data_vali["sft"][step]
    lbl_vali = data_vali['lbl'][step]
    confidences = sft_vali
    ground_truth = lbl_vali
    temperature = TemperatureScaling()

    histogram = HistogramBinning(hist_bins)
    iso = IsotonicRegression()
    bbq = BBQ()
    enir = ENIR()
    method = 'mle'

    lr_calibration = LogisticCalibration(detection=False, method=method)
    temperature = TemperatureScaling(detection=False, method=method)
    betacal = BetaCalibration(detection=False, method=method)

    models = [("hist", histogram),
              ("iso", iso),
              ("bbq", bbq),
              ("enir", enir),
              ("lr", lr_calibration),
              ("temperature", temperature),
              ("beta", betacal)]


    ace = ACE(bins)
    ece = ECE(bins)
    mce = MCE(bins)
    validation_set_sm = confidences
    validation_set_gt = ground_truth
    predictions = []
    all_ace = [ace.measure(validation_set_sm, validation_set_gt)]
    all_ece = [ece.measure(validation_set_sm, validation_set_gt)]
    all_mce = [mce.measure(validation_set_sm, validation_set_gt)]
    for model in models:
        name, instance = model
        print("Build %s model" % name)
        instance.fit(build_set_sm, build_set_gt)

    for model in models:
        _, instance = model
        prediction = instance.transform(validation_set_sm)
        predictions.append(prediction)

        all_ace.append(ace.measure(prediction, validation_set_gt))
        all_ece.append(ece.measure(prediction, validation_set_gt))
        all_mce.append(mce.measure(prediction, validation_set_gt))
    x = [0, 1, 2, 3, 4, 5, 6, 7]
    # plt.plot(x, all_ece, x, all_mce)  # 此时x不可省略
    plt.plot(x, all_ece, )  # 此时x不可省略

    plt.show()
    print(all_ece)
    print(all_mce)
    bins2 = np.linspace(0.1, 1, bins)

    diagram = ReliabilityDiagram(bins=bins, title_suffix="default")
    diagram.plot(validation_set_sm, validation_set_gt, filename="/home/mao/23Summer/code/racing-car/balanced20cnn/thr2-cali/"+str(step)+"test"+str(all_ece[0])+".png")

    method_num=np.argmin(all_ece)
    print(method_num)

    diagram = ReliabilityDiagram(bins=bins, title_suffix=models[method_num][0])
    prediction=predictions[method_num]
    diagram.plot(prediction, validation_set_gt, filename="/home/mao/23Summer/code/racing-car/balanced20cnn/thr2-cali/"+str(step)+"step"+str(method_num)+"-"+str(all_ece[method_num]) + ".png")

    binned = np.digitize(prediction, bins2)
    dset = []
    binsamount = []
    for i in range(10):
        posi=list(np.where(binned==i))[0]
        binsamount.append(len(posi))
        if len(posi)>30:
            dsubset=[]
            for im in range(200):
                temp_cali = []
                temp_gt = []
                for jm in range(1000):
                    inumber = np.random.randint(0,len(posi))
                    temp_cali.append(prediction[inumber])
                    temp_gt.append(validation_set_gt[inumber])
                temp_cali = np.array(temp_cali)
                temp_gt = np.array(temp_gt)
                mu_cali = np.mean(temp_cali)
                mu_gt = np.mean(temp_gt)
                dsubset.append(abs(mu_cali-mu_gt))
            dsubset_np=np.array(dsubset)
            dsubset_np=np.sort(dsubset_np)
            dset.append(dsubset_np)
        else:
            dset.append(0)

            # ki = (1-0.9)*(1+ni/2)
    print(binsamount)
    np.savez_compressed(str(step)+"k95200.npz",dset=dset,lbl=validation_set_gt,cali=prediction,ori=validation_set_sm,ece=all_ece,mce=all_mce,number=method_num)
    print("end")
    return all_ece[0],all_ece[method_num],all_ace[0],all_ace[method_num]



if __name__ == '__main__':
    testpath="rand-test-mono-cnn-action.npz"
    validpath="rand-valid-mono-cnn-action.npz"
    ece=[]
    ece2=[]
    mce=[]
    mce2=[]
    for i in range(0,20):
        print(i)
        a,b,c,d=main(i, testpath, validpath)
        ece2.append(b)
        mce2.append(d)
        ece.append(a)
        ece.append(c)
        print(ece)
        print(ece2)
        print(mce)
        print(mce2)
