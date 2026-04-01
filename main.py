import sys

import epics
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QPushButton, QLabel,
                             QLineEdit, QGroupBox, QFormLayout, QSpinBox, QButtonGroup)
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from collections import deque
import random
from copy import deepcopy
from epics import PV
from utils.find_peaks import get_spectrum, zero_pad_to_3x_power2
from datetime import datetime
import json
from utils.range_selector import RangeSelectorGraph
from utils.chrom_to_sext_response import CXY_TO_SEXT_CUR
from utils.generate_ramp import generate_base_ramp
from utils.array_processing import check_waveform_size


test_data = np.array([5.0158428499376244e-05, 1.043159235691089e-05, -4.796730755237623e-05, 3.909085338794059e-05, -4.407076246440595e-05, -3.156632048616832e-05, 6.89267221270693e-05, -4.220823904124753e-05, -1.2321525582277223e-05, 3.489093610267327e-05, -7.880565466349506e-05, 2.1971697396693065e-05, 5.002629414237622e-05, -6.840539723967328e-05, 2.5048618097663372e-05, -2.6775685274257538e-06, -6.856343669281189e-05, 6.266697019195049e-05, -4.536729888059412e-06, -5.0284297880920785e-05, 4.36516645429703e-05, -5.502188448241584e-05, -1.9153734893227718e-05, 7.073575697861386e-05, -5.907844741279206e-05, -8.176730784336637e-06, 2.9756064632079214e-05, -7.864486764940595e-05, 3.6163457080466344e-05, 3.65752317127277e-05, -7.28959191849505e-05, 2.7601948851386146e-05, -1.357747449910891e-05, -5.1013012064059395e-05, 6.726997212372277e-05, -2.3334074998524752e-05, -4.255546737935643e-05, 3.9738320413663366e-05, -5.534637749312871e-05, 3.922829363861388e-06, 5.9331209302475245e-05, -6.650509709377229e-05, 1.957866724267327e-06, 2.239332904619802e-05, -6.001372776353466e-05, 4.662062766620791e-05, 1.535646888079209e-05, -6.284551433982179e-05, 3.2964983558001975e-05, -1.6175366735544548e-05, -2.6025611618841597e-05, 5.8956862829643565e-05, -3.8759278725752486e-05, -2.5084215238553464e-05, 3.918301449211881e-05, -4.8656860177504945e-05, 1.7381328441287126e-05, 4.0419965034158427e-05, -6.702301521973266e-05, 1.4275038144148521e-05, 1.7432645378841594e-05, -4.833009654346534e-05, 4.625507991540595e-05, -3.7278420536633705e-06, -5.46291658594396e-05, 3.7792383814148514e-05, -2.3067559617940588e-05, -1.625917066435644e-05, 5.353944361752475e-05, -5.3428779548471284e-05, -1.7812716543940588e-05, 3.935993878207921e-05, -5.361992532336635e-05, 2.4658381385950494e-05, 3.2338621079336636e-05, -7.634388729132674e-05, 2.065698361048515e-05, 1.2230149625940591e-05, -4.781778886207427e-05, 5.448138011642871e-05, -1.8365855346633665e-05, -5.754337855029702e-05, 4.679771109041583e-05, -3.329924883459603e-05, -1.0120022221584161e-05, 5.907206741538613e-05, -7.082271702181187e-05, -1.242786016766337e-05, 4.48141847429703e-05, -6.185949492473269e-05, 3.3714150813564365e-05, 2.562885970947523e-05, -8.524625308117821e-05, 3.305594680428119e-05, 6.017858549069293e-06, -4.814047342809901e-05, 6.067228924683167e-05, -3.591442176228714e-05, -5.293268407393068e-05, 5.6643962222336626e-05, -4.4816833790297025e-05, -4.325725401960398e-06, 5.359310642201978e-05, -8.343863772658414e-05, 3.109315683168269e-07, 4.2171618108089106e-05, -6.427912351900989e-05, 3.796243255452475e-05, 7.581717194475251e-06, -8.052875320247524e-05, 4.376932533144555e-05, -4.5985556830413045e-06, -3.8381700712118826e-05, 5.486828857181189e-05, -5.144208248503962e-05, -3.406698688475246e-05, 5.717508585047226e-05, -4.8240448294415836e-05, 6.169395412980201e-06, 3.6882300580316825e-05, -7.844003655960395e-05, 1.9387474701683167e-05, 3.266034955560397e-05, -5.4142443925267325e-05, 3.7897352713386127e-05, -8.165508488702974e-06, -5.543691070198019e-05, 5.0791632924544544e-05, -1.4925089477970294e-05, -2.4254341338316833e-05, 4.457478696485149e-05, -5.195522739552179e-05, -7.889258312871285e-06, 4.997268069241583e-05, -5.017207461766336e-05, 1.2885187711188117e-05, 2.398919227170297e-05, -6.378935337415842e-05, 3.176186797400991e-05, 1.8220178752861398e-05, -4.837328961715843e-05, 3.6619918840891085e-05, -1.8281114885326733e-05, -3.893700848114851e-05, 5.1401369591910887e-05, -2.7947135200544552e-05, -1.800200101397029e-05, 3.9791898798337615e-05, -5.856637891282179e-05, 2.430642976831691e-06, 4.583089472633664e-05, -5.730011382490596e-05, 1.7081589715869003e-05, 1.5115147105247532e-05, -6.68118861018812e-05, 4.048235095125742e-05, 9.874312559207917e-06, -4.98390744720198e-05, 4.126100640653464e-05, -3.292043283001386e-05, -3.6532172122891097e-05, 6.159902551502971e-05, -4.0268902377008905e-05, -1.5173777888514835e-05, 4.197613258631683e-05, -7.234585183698018e-05, 1.1986669755792083e-05, 4.85548357297604e-05, -6.741889431990098e-05, 2.404179566253465e-05, 7.867377849069298e-06, -7.030114762814853e-05, 5.35307052181089e-05, -1.565532926771289e-06, -5.2130128582019805e-05, 4.813654022524555e-05, -4.611532539438613e-05, -2.8748660631751492e-05, 6.675736693769305e-05, -5.5881321645592074e-05, -1.0303096919336634e-05, 3.9495190849712865e-05, -7.76056758481188e-05, 2.356081274846535e-05, 3.814129516079209e-05, -7.227208047062375e-05, 2.9462821145445558e-05, -4.014827000891094e-06, -6.0289943529703e-05, 5.833392398329703e-05, -1.8644143424188107e-05, -4.3899242347039607e-05, 4.617988990643567e-05, -5.237292136344554e-05, -1.1114796971564364e-05, 5.783125099850495e-05, -6.17195192030792e-05, 1.313422989891089e-06, 2.977665067683169e-05, -6.569688122193067e-05, 3.3887977867336634e-05, 2.036453992851484e-05, -5.990155079237624e-05, 3.476688706158415e-05, -1.139540812011881e-05, -3.7391404436732675e-05, 5.311794763392078e-05, -3.0123886599643565e-05, -2.3793695584752484e-05, 4.2842504658534655e-05, -4.784357081091088e-05, 4.708774743960388e-06, 4.166567843762377e-05, -5.722519808782179e-05, 1.4883881819326738e-05, 2.156994070112872e-05, -5.22442941529901e-05, 3.548984555748317e-05, 4.176524906628713e-06, -4.632396226524752e-05, 3.7854788319664355e-05, -1.9380453069712875e-05, -2.556740589453466e-05, 4.72838577378416e-05, -3.985128211236139e-05, -1.3649998871713865e-05, 3.911568667396239e-05, -5.2478866958217816e-05, 1.1495287775564362e-05, 3.388637334435643e-05, -6.158387231480197e-05, 1.9651646430207938e-05, 1.300058076212871e-05, -5.23808477169109e-05, 4.238909315611881e-05, -7.4010458363495025e-06, -4.825418225118811e-05, 4.253128827950495e-05, -3.139793260442277e-05, -2.0698275027287132e-05, 5.425294899178217e-05, -5.475833458725742e-05, -1.1835342268415832e-05, 4.169139123366337e-05, -6.189734374937625e-05, 2.129794234837622e-05, 3.151437302420793e-05, -7.250664916535445e-05, 2.7892033007277233e-05, 6.6439328640594065e-06, -5.3762284687267315e-05, 5.285039948975246e-05, -2.1733424959623758e-05, -4.912863151027723e-05, 5.13441649479604e-05, -4.313319209036635e-05, -1.5022892596356441e-05, 5.531417761138614e-05, -6.847603813046931e-05, -4.086181437407431e-06, 3.986348957680199e-05, -6.686222012137624e-05, 2.8154141371100985e-05, 1.8991774986168326e-05, -7.294915586396038e-05, 3.5783304581386126e-05, -4.808052825324764e-06, -4.6953329865396035e-05, 5.27120791030594e-05, -3.588111103251485e-05, -3.722566784762377e-05, 4.996302399736437e-05, -4.904188105993069e-05, -3.792513053485145e-06, 4.4317954912544555e-05, -6.725689060435642e-05, 8.79573468953465e-06, 2.8975471407227716e-05, -5.722685724689109e-05, 3.271808716661583e-05, 5.430334207435651e-06, -5.417395665582179e-05, 3.905425659762375e-05, -1.405656794752475e-05, -2.9229587617722775e-05, 4.654020406900989e-05, -3.765090392566336e-05, -1.572688564941584e-05, 4.191984639314852e-05, -4.626987496957427e-05, 7.857906066336635e-06, 3.26271199598515e-05, -5.399171684207921e-05, 1.8936735623355446e-05, 1.658538683326734e-05, -4.520069948980199e-05, 3.37225426145198e-05, -4.00306669339604e-06, -3.7848346379762784e-05, 3.7681342277871275e-05, -2.273004488330198e-05, -1.864565533477228e-05, 4.146442046821782e-05, -4.22068075009604e-05, -6.791659098630696e-06, 3.586360792336635e-05, -4.8957949106743554e-05, 1.3090739974653457e-05, 2.4061494814554462e-05, -5.60704803020099e-05, 2.3579189624376242e-05, 8.507923763465339e-06, -4.446990049615843e-05, 3.7678071853118806e-05, -1.7246015827227726e-05, -3.8336488846643565e-05, 4.429098270386139e-05, -3.272532536772278e-05, -1.5932142409722774e-05, 4.367022455940595e-05, -5.7000151710673256e-05, -1.9428698667623803e-06, 3.983052969188118e-05, -5.755269026696039e-05, 1.9302975634009897e-05, 1.7592569523881187e-05, -6.452925455445544e-05, 3.529039661112171e-05, 2.0638163060791964e-06, -4.776712124632474e-05, 4.517770990394058e-05, -3.158990227404951e-05, -3.630627324892079e-05, 5.379770577777227e-05, -4.5008388513353454e-05, -1.3086572071217825e-05, 4.3400321193485143e-05, -6.720507825841584e-05, 7.004403360693067e-06, 3.619877825872278e-05, -6.363568220805938e-05, 2.4570436250019816e-05, 6.65469097846534e-06, -6.216920695115347e-05, 4.1833455996633666e-05, -1.0468761054736438e-05, -4.308172188069307e-05, 4.49992402830891e-05, -4.128003333354456e-05, -2.5373526919306933e-05, 4.969499230115843e-05, -5.017423331323762e-05, -3.8822884170891094e-06, 3.4356114642574264e-05, -6.239132168976237e-05, 1.5519930080881196e-05, 2.418251953960397e-05, -5.299532172851484e-05, 2.876379303387129e-05, -2.832596368514847e-06, -4.492156701376237e-05, 4.0222870955287114e-05, -1.7189558518554472e-05, -2.501436389340595e-05, 4.021529428265347e-05, -3.927460741423763e-05, -9.847141613623772e-06, 3.9494903573950496e-05, -4.284884213128712e-05, 7.855681299570308e-06, 2.4977692008613855e-05, -4.873566887558415e-05, 2.014079914482178e-05, 1.3553522534158423e-05, -3.815403276186138e-05, 2.9783134746801e-05, -9.61120751465347e-06, -3.108915867980197e-05, 3.666326927283169e-05, -2.2311824457128704e-05, -1.4802443852237624e-05, 3.490825312990099e-05, -4.116516311159406e-05, -2.8016588059306967e-06, 3.318766764801981e-05, -4.392696842066337e-05, 1.1279781019326733e-05, 1.6653637718653463e-05, -4.837574036202971e-05, 2.4902296879801978e-05, 4.741806038544555e-06, -3.9231307757891096e-05, 3.235633915215842e-05, -2.063787113277228e-05, -2.9211084667267333e-05, 4.271377127725743e-05, -3.4430378324742574e-05, -1.48567737570792e-05, 3.677886953475249e-05, -5.175850021623761e-05, 4.006386890465338e-06, 3.3715971427782167e-05, -5.5592882269702966e-05, 1.691096098217821e-05, 1.164238039126733e-05, -5.319382868083167e-05, 3.6464651010297016e-05, -6.044149448960397e-06, -4.478725087217227e-05, 4.1295772762376235e-05, -3.1625718485595044e-05, -2.6281629400287132e-05, 4.919408908750496e-05, -4.9055502286940594e-05, -1.1874417301663367e-05, 3.84179823751188e-05, -5.925513069157327e-05, 1.1407009047257424e-05, 2.6900323690792082e-05, -6.19979205467525e-05, 2.3754558488217824e-05, 3.047699290940589e-06, -5.15219282298812e-05, 4.052138163613762e-05, -1.8609045194811883e-05, -4.0086648064762365e-05, 4.22820485989901e-05, -3.912742995158315e-05, -1.881498502712871e-05, 4.3899189682069315e-05, -5.1691746878400106e-05, -3.6030027803960374e-06, 3.0219669659900997e-05, -5.465274688428713e-05, 1.6810334848831676e-05, 1.7254532220000003e-05, -5.0028057656949517e-05, 2.6036705818712876e-05, -5.025395492910891e-06, -3.730750018554457e-05, 3.86039455949802e-05, -2.019376922783577e-05, -2.4087543819435642e-05, 3.597036364940596e-05, -3.581322267972278e-05, -6.446055990297028e-06, 3.642005597049505e-05, -4.063394428918811e-05, 4.5755264636633696e-06, 2.1309901113811873e-05, -4.13208385190099e-05, 2.036700851616139e-05, 1.0739790012277229e-05, -3.540949328329702e-05, 2.512191348938614e-05, -9.517530290247522e-06, -2.5232130664376236e-05, 3.5364581935148516e-05, -2.2146030089059405e-05, -1.58869905840792e-05, 3.10450461060396e-05, -3.566787994551584e-05, -9.51826663841582e-07, 3.0125288116475245e-05, -4.082346348932673e-05, 7.610067747267329e-06, 1.4508342580502966e-05, -4.07378763282574e-05, 2.392808800480198e-05, 1.3200701472475266e-06, -3.6682266254728706e-05, 2.9149678638613856e-05, -1.9080126741871294e-05, -2.4562771126900994e-05, 3.907030455496039e-05, -3.4554498639108914e-05, -1.4538181051586132e-05, 3.4083693059188116e-05, -4.612620976772279e-05, 4.176128103059405e-06, 2.7553592255841577e-05, -5.159695179108909e-05, 1.6563347394118816e-05, 9.363563130099004e-06, -4.751442360425742e-05, 3.34441397357396e-05, -1.0571517812792069e-05, -3.987113920326734e-05, 4.038182508386138e-05, -3.12614633085594e-05, -2.4308901148326728e-05, 4.364401168993069e-05, -4.747826678722771e-05, -8.697891485900998e-06, 3.544392691415842e-05, -5.589470193882376e-05, 1.0150167900207925e-05, 2.0887000010297022e-05, -5.615182245475248e-05, 2.4715903382990097e-05, -3.251869612277322e-07, -4.8370149491772266e-05, 3.7705162091881176e-05, -2.070807770792079e-05, -3.579827863792078e-05, 4.0691829869742566e-05, -3.8991617334039594e-05, -1.799641876171287e-05, 4.003374637029703e-05, -4.894359378457426e-05, -3.321211594455448e-06, 2.7114063040851478e-05])
with open("booster_knobs.json", "r") as f:
    knobs = json.load(f)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout(pad=3.0)

        self.setFocusPolicy(Qt.StrongFocus)

        self.saved_xlim = None
        self.saved_ylim = None
        self.original_xlim = None
        self.original_ylim = None
        self.initial_zoom_saved = False

        self.pan_start_x = None
        self.pan_start_y = None
        self.pan_start_xlim = None
        self.pan_start_ylim = None
        self.panning = False

        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)

        self.add_reset_button()

    def add_reset_button(self):
        from PyQt5.QtWidgets import QPushButton

        self.reset_button = QPushButton('Reset', self)

        reset_button_style = """
            QPushButton {
                background-color: #6c757d;
                border: none;
                color: white;
                padding: 5px 10px;
                font-size: 10px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #545b62;
            }
        """
        self.reset_button.setStyleSheet(reset_button_style)

        self.reset_button.setFixedSize(60, 28)

        self.reset_button.clicked.connect(self.on_reset_click)

        self.reset_button.move(10, self.height() - 40)

    def on_reset_click(self):
        self.reset_zoom()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'reset_button'):
            self.reset_button.move(10, self.height() - 40)

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata

        if xdata is None or ydata is None:
            return

        if event.button == 'up':
            scale_factor = 0.9
        else:
            scale_factor = 1.1

        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_xlim = (xdata - new_width * (xdata - xlim[0]) / (xlim[1] - xlim[0]),
                    xdata + new_width * (xlim[1] - xdata) / (xlim[1] - xlim[0]))

        new_height = (ylim[1] - ylim[0]) * scale_factor
        new_ylim = (ydata - new_height * (ydata - ylim[0]) / (ylim[1] - ylim[0]),
                    ydata + new_height * (ylim[1] - ydata) / (ylim[1] - ylim[0]))

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)

        self.saved_xlim = new_xlim
        self.saved_ylim = new_ylim

        self.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 3:
            return

        self.panning = True
        self.pan_start_x = event.xdata
        self.pan_start_y = event.ydata
        self.pan_start_xlim = self.ax.get_xlim()
        self.pan_start_ylim = self.ax.get_ylim()

        self.setCursor(Qt.ClosedHandCursor)

    def on_motion(self, event):
        if not self.panning or event.inaxes != self.ax:
            return

        dx = event.xdata - self.pan_start_x
        dy = event.ydata - self.pan_start_y

        xlim = (self.pan_start_xlim[0] - dx, self.pan_start_xlim[1] - dx)
        ylim = (self.pan_start_ylim[0] - dy, self.pan_start_ylim[1] - dy)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        self.saved_xlim = xlim
        self.saved_ylim = ylim

        self.draw_idle()

    def on_release(self, event):
        self.panning = False
        self.setCursor(Qt.ArrowCursor)

    def save_current_zoom(self):
        self.saved_xlim = self.ax.get_xlim()
        self.saved_ylim = self.ax.get_ylim()

    def restore_zoom(self):
        if self.saved_xlim is not None and self.saved_ylim is not None:
            self.ax.set_xlim(self.saved_xlim)
            self.ax.set_ylim(self.saved_ylim)

    def save_original_zoom(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        if xlim != (0.0, 1.0) or ylim != (0.0, 1.0):
            self.original_xlim = xlim
            self.original_ylim = ylim
            self.initial_zoom_saved = True

    def reset_zoom(self):
        if self.original_xlim is not None and self.original_ylim is not None:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.saved_xlim = self.original_xlim
            self.saved_ylim = self.original_ylim
            self.draw_idle()
        else:
            self.ax.autoscale()
            self.saved_xlim = self.ax.get_xlim()
            self.saved_ylim = self.ax.get_ylim()
            self.draw_idle()


class DataManager(QObject):
    data_updated = pyqtSignal()
    sextupole_updated = pyqtSignal(str)

    def __init__(self, knobs):
        super().__init__()
        self.data = {"x": [np.nan], "cx": np.nan, "qx": np.nan, "qx_m": np.nan, "qx_p": np.nan, "freq_spectrum_qx": [np.nan], "amplitude_spectrum_qx": [np.nan],
                     "y": [np.nan], "cy": np.nan, "qy": np.nan, "qy_m": np.nan, "qy_p": np.nan, "freq_spectrum_qy": [np.nan], "amplitude_spectrum_qy": [np.nan],
                     "qs": np.nan, "sigma_E_spread": knobs["constants"]["sigma_E_spread"], "energy": knobs["constants"]["energy"], "turns": knobs["constants"]["turns"],
                     "skip_turns": knobs["constants"]["skip_turns"]}

        self.max_length = 100
        self.freq_range_qx = [0.3, 0.41]
        self.freq_range_qy = [0.3, 0.41]
        self.freq_range_qs = [0.01, 0.1]
        self.cx_array = deque(maxlen=self.max_length)
        self.cy_array = deque(maxlen=self.max_length)

        self.cxy_inv = np.array(CXY_TO_SEXT_CUR)

        self.bpm_to_observe = knobs["constants"]["bpm_to_observe"]
        self.bpm_x_pv = self.bpm_y_pv = None
        self.initial_sextupole_set_values = {}
        self.initial_sextupole_set_waveform_values = {}
        self.sextupole_set_values = {}
        self.sextupole_set_pvs = {}
        self.sextupole_name_to_pvset_name_map = {}
        self.sextupole_meas_values = {}
        self.sextupole_meas_pvs = {}
        self.sextupole_name_to_pvmeas_name_map = {}
        self.initialize_bpm_pv(self.bpm_to_observe)
        self.initialize_sextupole_pv()
        self.initial_chrom_values = [0, 0]

    def initialize_bpm_pv(self, bpm_to_observe):
        if self.bpm_x_pv is not None:
            self.bpm_x_pv.remove_callback(0)
            self.bpm_y_pv.remove_callback(0)
            self.bpm_x_pv.disconnect()
            self.bpm_y_pv.disconnect()
        pvx = knobs["bpm"][bpm_to_observe]["turns_h"]
        pvy = knobs["bpm"][bpm_to_observe]["turns_v"]
        self.pvs_updated = {pvx: False, pvy: False}
        self.pvs_to_plane_map = {pvx: "x", pvy: "y"}
        self.bpm_x_pv = PV(pvx, callback=self.update_data_callback)
        self.bpm_x_pv.connect()
        self.bpm_y_pv = PV(pvy, callback=self.update_data_callback)
        self.bpm_y_pv.connect()

    def initialize_sextupole_pv(self):
        for sext_name, sext_config in knobs["sextupole"].items():
            pv_set_name = sext_config["set_curr"]
            pv_set = PV(pv_set_name, callback=self.sextupole_callback)
            pv_set.connect()
            self.sextupole_set_pvs[pv_set_name] = pv_set
            self.sextupole_set_values[pv_set_name] = pv_set.value
            self.sextupole_name_to_pvset_name_map[sext_name] = pv_set_name
            self.sextupole_updated.emit(pv_set_name)
            pv_meas_name = sext_config["meas_curr"]
            pv_meas = PV(pv_meas_name, callback=self.sextupole_callback)
            pv_meas.connect()
            self.sextupole_meas_pvs[pv_meas_name] = pv_meas
            self.sextupole_meas_values[pv_meas_name] = pv_meas.value
            self.sextupole_name_to_pvmeas_name_map[sext_name] = pv_meas_name
            self.sextupole_updated.emit(pv_meas_name)

        self.initial_sextupole_set_values = deepcopy(self.sextupole_set_values)

        for sext_name, sext_config in knobs["sextupole"].items():
            pv_set_waveform_name = sext_config["set_waveform"]
            val = epics.caget(pv_set_waveform_name)
            val = check_waveform_size(val, knobs["constants"]["waveform_length"])
            self.initial_sextupole_set_waveform_values[pv_set_waveform_name] = val

    def update_data_callback(self, pvname=None, value=None, **kw):
        if not self.pvs_updated[pvname]:
            plane = self.pvs_to_plane_map[pvname]
            if isinstance(value, float):
                value = [value]
            if not len(value):
                value = [np.nan]
            self.data[plane] = value - np.nanmean(value)
            # self.data[plane] = test_data - np.nanmean(test_data)
            # self.data[plane] += np.max(self.data[plane]) * 1e-1 * np.random.random(len(test_data))
            self.pvs_updated[pvname] = True

        for pv in self.pvs_updated:
            if not self.pvs_updated[pv]:
                return

        self.analyze_data(self.data["x"][self.data["skip_turns"]:], self.data["y"][self.data["skip_turns"]:])

        self.data_updated.emit()

        for pv in self.pvs_updated:
            self.pvs_updated[pv] = False

    def sextupole_callback(self, pvname=None, value=None, **kw):
        if pvname in self.sextupole_set_pvs:
            self.sextupole_set_values[pvname] = value
        if pvname in self.sextupole_meas_pvs:
            self.sextupole_meas_values[pvname] = value
        self.sextupole_updated.emit(pvname)

    def _find_satellites_peaks(self, q0, qs, freq_spectrum, amplitude_spectrum):
        ampl_q0_minus_satellite = 0
        ampl_q0_plus_satellite = 0
        idx_q0_minus_satellite = 0
        idx_q0_plus_satellite = 0
        if len(freq_spectrum) > 0:
            idx_q0_minus_satellite = np.argmin(np.abs(freq_spectrum - (q0 - qs)))
            if 0 < idx_q0_minus_satellite < len(freq_spectrum) - 1:
                neighbors = [idx_q0_minus_satellite - 1, idx_q0_minus_satellite, idx_q0_minus_satellite + 1]
                ampl_q0_minus_satellite = np.max(amplitude_spectrum[neighbors])
            idx_q0_plus_satellite = np.argmin(np.abs(freq_spectrum - (q0 + qs)))
            if 0 < idx_q0_plus_satellite < len(freq_spectrum) - 1:
                neighbors = [idx_q0_plus_satellite - 1, idx_q0_plus_satellite, idx_q0_plus_satellite + 1]
                ampl_q0_plus_satellite = np.max(amplitude_spectrum[neighbors])
        return idx_q0_minus_satellite, ampl_q0_minus_satellite, idx_q0_plus_satellite, ampl_q0_plus_satellite

    def analyze_data(self, x, y):
        if len(x) >= self.data["turns"]:
            x = zero_pad_to_3x_power2(x)
            y = zero_pad_to_3x_power2(y)
            qx, qx_amplitude, freq_spectrum_qx, amplitude_spectrum_qx,  = get_spectrum(x, self.data["turns"], self.freq_range_qx)
            qy, qy_amplitude, freq_spectrum_qy, amplitude_spectrum_qy = get_spectrum(y, self.data["turns"], self.freq_range_qy)
            qs, _, _, _ = get_spectrum(x, self.data["turns"], self.freq_range_qs)

            idx_qx_minus_satellite, ampl_qx_minus_satellite, idx_qx_minus_satellite, ampl_qx_plus_satellite = self._find_satellites_peaks(qx, qs, freq_spectrum_qx, amplitude_spectrum_qx)
            idx_qy_minus_satellite, ampl_qy_minus_satellite, idx_qy_minus_satellite, ampl_qy_plus_satellite = self._find_satellites_peaks(qy, qs, freq_spectrum_qy, amplitude_spectrum_qy)

            self.data["cx"] = qs / self.data["sigma_E_spread"] * np.sqrt((ampl_qx_plus_satellite + ampl_qx_minus_satellite) / qx_amplitude)
            self.data["cy"] = qs / self.data["sigma_E_spread"] * np.sqrt((ampl_qy_plus_satellite + ampl_qy_minus_satellite) / qy_amplitude)
            self.data["qs"] = qs

            self.data["qx"] = qx
            self.data["qx_m"] = qx - qs
            self.data["qx_p"] = qx + qs
            self.data["freq_spectrum_qx"] = freq_spectrum_qx
            self.data["amplitude_spectrum_qx"] = amplitude_spectrum_qx

            self.data["qy"] = qy
            self.data["qy_m"] = qy - qs
            self.data["qy_p"] = qy + qs
            self.data["freq_spectrum_qy"] = freq_spectrum_qy
            self.data["amplitude_spectrum_qy"] = amplitude_spectrum_qy

            self.cx_array.append(self.data["cx"])
            self.cy_array.append(self.data["cy"])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chromaticity measurement")
        self.setGeometry(100, 100, 1400, 900)

        self.data_manager = DataManager(knobs)

        self.data_manager.data_updated.connect(self.update_plots)
        self.data_manager.sextupole_updated.connect(self.fill_sextupoles_field)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        plots_layout = QGridLayout()

        self.canvases = []
        for i in range(6):
            canvas = MplCanvas(self, width=5, height=4, dpi=100)
            self.canvases.append(canvas)
            row = i // 3
            col = i % 3
            plots_layout.addWidget(canvas, row, col)

        self.init_range_selectors()

        buttons_layout = QGridLayout()
        self.button_group = QButtonGroup()
        self.button_group.setExclusive(True)
        button_style = """
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:checked {
                background-color: #ff5722;
                border: 2px solid #ff9800;
            }
        """
        bpms = list(knobs["bpm"].keys())
        bpm_to_observe = knobs["constants"]["bpm_to_observe"]
        self.buttons = []
        for idx, bpm in enumerate(bpms):
            btn = QPushButton(bpm)
            btn.setStyleSheet(button_style)
            btn.setCheckable(True)
            if bpm == bpm_to_observe:
                btn.setChecked(True)
            else:
                btn.setChecked(False)
            btn.clicked.connect(self.choose_bpm)
            self.buttons.append(btn)
            self.button_group.addButton(btn)
            if idx < len(bpms) / 2:
                buttons_layout.addWidget(btn, 0, idx)
            else:
                buttons_layout.addWidget(btn, 1, idx - len(bpms) // 2)

        text_field_style = """
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 6px 10px;
                font-family: 'Arial', sans-serif;
                font-size: 14px;
                color: #000000;
            }
            QLineEdit:focus {
                border-color: #80bdff;
                background-color: #ffffff;
                outline: 0;
                color: #000000;
            }
            QLineEdit:read-only {
                background-color: #f5f5f5;
                color: #000000;
            }
        """
        right_panel_layout = QHBoxLayout()
        fields_group = QGroupBox("Parameters")
        fields_group.setFont(QFont("Arial", 10, QFont.Bold))
        fields_group.setMaximumWidth(450)
        fields_layout = QFormLayout()
        fields_layout.setSpacing(5)
        fields_layout.setContentsMargins(10, 10, 10, 10)

        self.field1 = QLineEdit()
        self.field2 = QLineEdit()
        self.field3 = QLineEdit()
        self.field4 = QLineEdit()
        self.field5 = QLineEdit()
        self.field6 = QLineEdit()
        self.field7 = QLineEdit()

        self.field1.setStyleSheet(text_field_style)
        self.field2.setStyleSheet(text_field_style)
        self.field3.setStyleSheet(text_field_style)
        self.field4.setStyleSheet(text_field_style)
        self.field5.setStyleSheet(text_field_style)
        self.field6.setStyleSheet(text_field_style)
        self.field7.setStyleSheet(text_field_style)

        fields_layout.addRow("SigmaE/E:", self.field1)
        fields_layout.addRow("Energy (Gev):", self.field2)
        fields_layout.addRow("Turns:", self.field3)
        fields_layout.addRow("Skip turns:", self.field4)
        fields_layout.addRow("Qs (manual):", self.field5)
        fields_layout.addRow("Qx (manual):", self.field6)
        fields_layout.addRow("Qy (manual):", self.field7)

        self.field1.setReadOnly(False)
        self.field1.setText(str(self.data_manager.data["sigma_E_spread"]))
        self.field2.setReadOnly(False)
        self.field2.setText(str(self.data_manager.data["energy"]))
        self.field3.setReadOnly(False)
        self.field3.setText(str(int(self.data_manager.data["turns"])))
        self.field4.setReadOnly(False)
        self.field4.setText(str(int(self.data_manager.data["skip_turns"])))
        self.field5.setReadOnly(False)
        self.field6.setReadOnly(False)
        self.field7.setReadOnly(False)

        self.field1.returnPressed.connect(self.update_fields)
        self.field2.returnPressed.connect(self.update_fields)
        self.field3.returnPressed.connect(self.update_fields)
        self.field4.returnPressed.connect(self.update_fields)
        self.field5.returnPressed.connect(self.update_fields)
        self.field6.returnPressed.connect(self.update_fields)
        self.field7.returnPressed.connect(self.update_fields)

        fields_group.setLayout(fields_layout)

        control_buttons_group = QGroupBox("Knobs")
        control_buttons_group.setFont(QFont("Arial", 10, QFont.Bold))
        control_buttons_group.setMaximumWidth(300)
        control_buttons_layout = QVBoxLayout()
        control_buttons_layout.setSpacing(10)

        self.read_btn = QPushButton("Read")
        self.read_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                border: none;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.read_btn.clicked.connect(self.on_read_button_clicked)
        control_buttons_layout.addWidget(self.read_btn)

        spinbox_buttons_layout = QVBoxLayout()

        spinbox_group1 = QWidget()
        spinbox_group1_layout = QHBoxLayout()
        spinbox_group1_layout.setContentsMargins(0, 0, 0, 0)

        self.spinbox1_label = QLabel("Cx")
        self.spinbox1_label.setFont(QFont("Arial", 10))

        self.spinbox1_minus = QPushButton("-")
        self.spinbox1_minus.setFixedSize(30, 25)
        self.spinbox1_minus.clicked.connect(lambda: self.change_spinbox_value(1, -0.1))

        self.spinbox1_value = QLineEdit("0.0")
        self.spinbox1_value.setFixedWidth(60)
        self.spinbox1_value.setAlignment(Qt.AlignCenter)
        self.spinbox1_value.returnPressed.connect(lambda: self.change_spinbox_textfield_value(1))

        self.spinbox1_plus = QPushButton("+")
        self.spinbox1_plus.setFixedSize(30, 25)
        self.spinbox1_plus.clicked.connect(lambda: self.change_spinbox_value(1, 0.1))

        spinbox_group1_layout.addWidget(self.spinbox1_label)
        spinbox_group1_layout.addWidget(self.spinbox1_minus)
        spinbox_group1_layout.addWidget(self.spinbox1_value)
        spinbox_group1_layout.addWidget(self.spinbox1_plus)
        spinbox_group1.setLayout(spinbox_group1_layout)

        spinbox_group2 = QWidget()
        spinbox_group2_layout = QHBoxLayout()
        spinbox_group2_layout.setContentsMargins(0, 0, 0, 0)

        self.spinbox2_label = QLabel("Cy:")
        self.spinbox2_label.setFont(QFont("Arial", 10))

        self.spinbox2_minus = QPushButton("-")
        self.spinbox2_minus.setFixedSize(30, 25)
        self.spinbox2_minus.clicked.connect(lambda: self.change_spinbox_value(2, -0.1))

        self.spinbox2_value = QLineEdit("0.0")
        self.spinbox2_value.setFixedWidth(60)
        self.spinbox2_value.setAlignment(Qt.AlignCenter)
        self.spinbox2_value.returnPressed.connect(lambda: self.change_spinbox_textfield_value(2))

        self.spinbox2_plus = QPushButton("+")
        self.spinbox2_plus.setFixedSize(30, 25)
        self.spinbox2_plus.clicked.connect(lambda: self.change_spinbox_value(2, 0.1))

        spinbox_group2_layout.addWidget(self.spinbox2_label)
        spinbox_group2_layout.addWidget(self.spinbox2_minus)
        spinbox_group2_layout.addWidget(self.spinbox2_value)
        spinbox_group2_layout.addWidget(self.spinbox2_plus)
        spinbox_group2.setLayout(spinbox_group2_layout)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                border: none;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.reset_btn.clicked.connect(self.on_reset_button_clicked)

        spinbox_buttons_layout.addWidget(spinbox_group1)
        spinbox_buttons_layout.addWidget(self.reset_btn)
        spinbox_buttons_layout.addWidget(spinbox_group2)

        control_buttons_layout.addLayout(spinbox_buttons_layout)

        self.ramp_btn = QPushButton("Make Ramp")
        self.ramp_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                border: none;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.ramp_btn.clicked.connect(self.on_make_ramp_button_clicked)
        control_buttons_layout.addWidget(self.ramp_btn)

        control_buttons_group.setLayout(control_buttons_layout)

        sextupoles_fields_group = QGroupBox("Sextupoles")
        sextupoles_fields_group.setFont(QFont("Arial", 10, QFont.Bold))
        sextupoles_fields_layout = QGridLayout()
        sextupoles_fields_layout.setSpacing(8)

        button_style = """
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:checked {
                background-color: #ff5722;
                border: 2px solid #ff9800;
            }
        """

        self.sextupole_buttons = []
        self.sextupole_set_fields = {}
        self.sextupole_meas_fields = {}
        for idx, (sext, sext_data) in enumerate(knobs["sextupole"].items()):
            btn = QPushButton(sext)
            btn.setStyleSheet(button_style)
            # btn.setFixedSize(60, 50)
            self.sextupole_buttons.append(btn)
            sextupoles_fields_layout.addWidget(btn, 0, idx)

            text_field = QLineEdit()
            text_field.setPlaceholderText(f"Set")
            text_field.setReadOnly(True)
            text_field.setStyleSheet(text_field_style)
            self.sextupole_set_fields[sext_data["set_curr"]] = text_field
            sextupoles_fields_layout.addWidget(text_field, 1, idx)

            text_field = QLineEdit()
            text_field.setPlaceholderText(f"Meas")
            text_field.setReadOnly(True)
            text_field.setStyleSheet(text_field_style)
            self.sextupole_meas_fields[sext_data["meas_curr"]] = text_field
            sextupoles_fields_layout.addWidget(text_field, 2, idx)

        sextupoles_fields_group.setLayout(sextupoles_fields_layout)

        right_panel_layout.addWidget(fields_group)
        right_panel_layout.addWidget(control_buttons_group)
        right_panel_layout.addWidget(sextupoles_fields_group)
        right_panel_layout.addStretch()

        main_layout.addLayout(plots_layout, 7)
        main_layout.addLayout(buttons_layout, 1)
        main_layout.addLayout(right_panel_layout, 2)

    def init_range_selectors(self):
        self.x_tune_range_selector = RangeSelectorGraph(
            self.canvases[1].ax,
            orientation='horizontal',
            color='green'
        )

        # self.s_tune_range_selector = RangeSelectorGraph(
        #     self.canvases[1].ax,
        #     orientation='horizontal',
        #     color='yellow'
        # )

        self.y_tune_range_selector = RangeSelectorGraph(
            self.canvases[4].ax,
            orientation='horizontal',
            color='green'
        )

        self.x_tune_range_selector.setBounds(
            self.data_manager.freq_range_qx[0],
            self.data_manager.freq_range_qx[1]
        )
        # self.s_tune_range_selector.setBounds(
        #     self.data_manager.freq_range_qs[0],
        #     self.data_manager.freq_range_qs[1]
        # )
        self.y_tune_range_selector.setBounds(
            self.data_manager.freq_range_qy[0],
            self.data_manager.freq_range_qy[1]
        )

        self.x_tune_range_selector.span_selector.connect_event('button_release_event',
                                                               self.on_x_range_changed)
        # self.s_tune_range_selector.span_selector.connect_event('button_release_event',
        #                                                        self.on_s_range_changed)
        self.y_tune_range_selector.span_selector.connect_event('button_release_event',
                                                               self.on_y_range_changed)

    def on_x_range_changed(self, event):
        min_val, max_val = self.x_tune_range_selector.getRangeCoordinates()
        self.data_manager.freq_range_qx = [min_val, max_val]

    # def on_s_range_changed(self, event):
    #     min_val, max_val = self.s_tune_range_selector.getRangeCoordinates()
    #     self.data_manager.freq_range_qs = [min_val, max_val]

    def on_y_range_changed(self, event):
        min_val, max_val = self.y_tune_range_selector.getRangeCoordinates()
        self.data_manager.freq_range_qy = [min_val, max_val]

    def on_reset_button_clicked(self):
        pvs = list(self.data_manager.initial_sextupole_set_values.keys())
        values = list(self.data_manager.initial_sextupole_set_values.values())
        epics.caput_many(pvs, values)
        self.data_manager.initial_chrom_values = [0, 0]
        self.spinbox1_value.setText(f"{0:.2f}")
        self.spinbox2_value.setText(f"{0:.2f}")

        for pv, values in self.data_manager.initial_sextupole_set_waveform_values.items():
            values = np.clip(values, -knobs["constants"]["max_sextupole_current"], knobs["constants"]["max_sextupole_current"])
            epics.caput(pv, values)

    def on_make_ramp_button_clicked(self):
        base_ramp = generate_base_ramp(0.2, 3, knobs["constants"]["waveform_length"])
        for sext_name, sext_config in knobs["sextupole"].items():
            pv_set_waveform_name = sext_config["set_waveform"]
            pv_set_name = sext_config["set_curr"]
            init_waveform_values = self.data_manager.initial_sextupole_set_waveform_values[pv_set_waveform_name]
            waveform_values = self.data_manager.sextupole_set_values[pv_set_name] * base_ramp + init_waveform_values
            waveform_values = np.clip(waveform_values, -knobs["constants"]["max_sextupole_current"], knobs["constants"]["max_sextupole_current"])
            epics.caput(pv_set_waveform_name, waveform_values)
            epics.caput(pv_set_name, 0)

    def on_read_button_clicked(self):
        self.data_manager.initial_sextupole_set_values = deepcopy(self.data_manager.sextupole_set_values)
        self.data_manager.initial_chrom_values = [0, 0]
        self.spinbox1_value.setText(f"{0:.2f}")
        self.spinbox2_value.setText(f"{0:.2f}")

        for sext_name, sext_config in knobs["sextupole"].items():
            pv_set_name = sext_config["set_waveform"]
            self.data_manager.initial_sextupole_set_waveform_values[pv_set_name] = epics.caget(pv_set_name)

    def fill_sextupoles_field(self, pvname):
        if pvname in self.sextupole_set_fields:
            text_field = self.sextupole_set_fields[pvname]
            text_field.setText(f"{self.data_manager.sextupole_set_values[pvname]:.3f}")
        if pvname in self.sextupole_meas_fields:
            text_field = self.sextupole_meas_fields[pvname]
            text_field.setText(f"{self.data_manager.sextupole_meas_values[pvname]:.3f}")

    def change_spinbox_value(self, spinbox_id, delta):
        if spinbox_id == 1:
            self.calculte_and_set_sextupole_dI([delta, 0])
            current_value = float(self.spinbox1_value.text())
            new_value = current_value + delta
            self.spinbox1_value.setText(f"{new_value:.2f}")
        elif spinbox_id == 2:
            self.calculte_and_set_sextupole_dI([0, delta])
            current_value = float(self.spinbox2_value.text())
            new_value = current_value + delta
            self.spinbox2_value.setText(f"{new_value:.2f}")

    def change_spinbox_textfield_value(self, spinbox_id):
        if spinbox_id == 1:
            previous_value = self.data_manager.initial_chrom_values[0]
            current_value = float(self.spinbox1_value.text())
            delta = current_value - previous_value
            self.calculte_and_set_sextupole_dI([delta, 0])
            self.data_manager.initial_chrom_values[0] = current_value
        elif spinbox_id == 2:
            previous_value = self.data_manager.initial_chrom_values[1]
            current_value = float(self.spinbox2_value.text())
            delta = current_value - previous_value
            self.calculte_and_set_sextupole_dI([0, delta])
            self.data_manager.initial_chrom_values[1] = current_value

    def calculte_and_set_sextupole_dI(self, delta_cxy):
        coeff = self.data_manager.data["energy"] / 0.2
        dI_sext = self.data_manager.cxy_inv.dot(delta_cxy) * coeff
        pvs = list(self.data_manager.sextupole_set_values.keys())
        vals = np.array(list(self.data_manager.sextupole_set_values.values()))
        epics.caput_many(pvs, vals + dI_sext)

    def update_fields(self):
        self.data_manager.data["sigma_E_spread"] = abs(float(self.field1.text()))
        self.data_manager.data["energy"] = abs(float(self.field2.text()))
        self.data_manager.data["turns"] = abs(int(float(self.field3.text())))
        self.data_manager.data["skip_turns"] = abs(int(float(self.field4.text())))

    def update_plots(self):
        self.canvases[0].ax.clear()
        x = np.arange(1, len(self.data_manager.data["x"]) + 1, 1)
        y = self.data_manager.data["x"]
        self.canvases[0].ax.plot(x, y, 'b-', linewidth=2)
        self.canvases[0].ax.set_title(self.data_manager.bpm_to_observe)
        self.canvases[0].ax.axvline(self.data_manager.data["turns"], color='black', linestyle='--', alpha=0.5)
        self.canvases[0].ax.axvline(self.data_manager.data["skip_turns"], color='black', linestyle='--', alpha=0.5)
        self.canvases[0].ax.annotate(f'Turns = {self.data_manager.data["turns"]}',
                                     xy=(self.data_manager.data["turns"], max(y) * 0.7),
                                     xytext=(self.data_manager.data["turns"] + 5, max(y) * 0.7),
                                     fontsize=10, color='black', alpha=0.7,
                                     arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        self.canvases[0].ax.annotate(f'Start = {self.data_manager.data["skip_turns"]}',
                                     xy=(self.data_manager.data["skip_turns"], max(y) * -0.3),
                                     xytext=(self.data_manager.data["skip_turns"] + 5, max(y) * -0.3),
                                     fontsize=10, color='black', alpha=0.7,
                                     arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        self.canvases[0].ax.set_xlabel('Turn')
        self.canvases[0].ax.set_ylabel('X')
        self.canvases[0].ax.grid(True, alpha=0.3)

        self.canvases[1].ax.clear()
        x = self.data_manager.data["freq_spectrum_qx"]
        y = self.data_manager.data["amplitude_spectrum_qx"]
        qx = self.data_manager.data["qx"]
        qs = self.data_manager.data["qs"]
        qx_m = self.data_manager.data["qx_m"]
        qx_p = self.data_manager.data["qx_p"]
        self.canvases[1].ax.plot(x, y, 'b-', linewidth=2)
        self.canvases[1].ax.set_title(self.data_manager.bpm_to_observe)
        self.canvases[1].ax.set_xlabel('Tune Qx')
        self.canvases[1].ax.set_ylabel('Amplitude')
        self.canvases[1].ax.axvline(qx, color='black', linestyle='--', alpha=0.5)
        self.canvases[1].ax.annotate(f'Qx = {qx:.3f}',
                                     xy=(qx, max(y) * 0.9),
                                     xytext=(qx + 0.05, max(y) * 0.9),
                                     fontsize=10, color='black', alpha=0.7,
                                     arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        self.canvases[1].ax.axvline(qx_m, color='black', linestyle='--', alpha=0.5)
        self.canvases[1].ax.annotate(f'Qx-s = {qx_m:.3f}',
                                     xy=(qx_m, max(y) * 0.7),
                                     xytext=(qx_m - 0.15, max(y) * 0.7),
                                     fontsize=10, color='black', alpha=0.7,
                                     arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        self.canvases[1].ax.axvline(qx_p, color='black', linestyle='--', alpha=0.5)
        self.canvases[1].ax.annotate(f'Qx+s = {qx_p:.3f}',
                                     xy=(qx_p, max(y) * 0.5),
                                     xytext=(qx_p + 0.05, max(y) * 0.5),
                                     fontsize=10, color='black', alpha=0.7,
                                     arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        self.canvases[1].ax.axvline(qs, color='black', linestyle='--', alpha=0.5)
        self.canvases[1].ax.annotate(f'Qs = {qs:.3f}',
                                     xy=(qs, max(y) * 0.3),
                                     xytext=(qs + 0.05, max(y) * 0.3),
                                     fontsize=10, color='black', alpha=0.7,
                                     arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        self.canvases[1].ax.grid(True, alpha=0.3)

        self.canvases[2].ax.clear()
        x = np.arange(1, len(self.data_manager.cx_array) + 1, 1)
        y = self.data_manager.cx_array
        self.canvases[2].ax.plot(x, y, 'bo-', linewidth=2)
        self.canvases[2].ax.set_title(self.data_manager.bpm_to_observe)
        self.canvases[2].ax.set_xlabel('Count')
        self.canvases[2].ax.set_ylabel('Chromaticity Cx')
        self.canvases[2].ax.grid(True, alpha=0.3)

        self.canvases[3].ax.clear()
        x = np.arange(1, len(self.data_manager.data["y"]) + 1, 1)
        y = self.data_manager.data["y"]
        self.canvases[3].ax.plot(x, y, 'r-', linewidth=2)
        self.canvases[3].ax.set_title(self.data_manager.bpm_to_observe)
        self.canvases[3].ax.axvline(self.data_manager.data["turns"], color='black', linestyle='--', alpha=0.5)
        self.canvases[3].ax.axvline(self.data_manager.data["skip_turns"], color='black', linestyle='--', alpha=0.5)
        self.canvases[3].ax.annotate(f'Turns = {self.data_manager.data["turns"]}',
                                     xy=(self.data_manager.data["turns"], max(y) * 0.7),
                                     xytext=(self.data_manager.data["turns"] + 5, max(y) * 0.7),
                                     fontsize=10, color='black', alpha=0.7,
                                     arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        self.canvases[3].ax.annotate(f'Start = {self.data_manager.data["skip_turns"]}',
                                     xy=(self.data_manager.data["skip_turns"], max(y) * -0.3),
                                     xytext=(self.data_manager.data["skip_turns"] + 5, max(y) * -0.3),
                                     fontsize=10, color='black', alpha=0.7,
                                     arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        self.canvases[3].ax.set_xlabel('Turn')
        self.canvases[3].ax.set_ylabel('Y')
        self.canvases[3].ax.grid(True, alpha=0.3)

        self.canvases[4].ax.clear()
        x = self.data_manager.data["freq_spectrum_qy"]
        y = self.data_manager.data["amplitude_spectrum_qy"]
        qy = self.data_manager.data["qy"]
        qs = self.data_manager.data["qs"]
        qy_m = self.data_manager.data["qy_m"]
        qy_p = self.data_manager.data["qy_p"]
        self.canvases[4].ax.plot(x, y, 'r-', linewidth=2)
        self.canvases[4].ax.set_title(self.data_manager.bpm_to_observe)
        self.canvases[4].ax.set_xlabel('Tune Qy')
        self.canvases[4].ax.set_ylabel('Amplitude')
        self.canvases[4].ax.axvline(qy, color='black', linestyle='--', alpha=0.5)
        self.canvases[4].ax.annotate(f'Qy = {qy:.3f}',
                                     xy=(qy, max(y) * 0.9),
                                     xytext=(qy + 0.05, max(y) * 0.9),
                                     fontsize=10, color='black', alpha=0.7,
                                     arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        self.canvases[4].ax.axvline(qy_m, color='black', linestyle='--', alpha=0.5)
        self.canvases[4].ax.annotate(f'Qy-s = {qy_m:.3f}',
                                     xy=(qy_m, max(y) * 0.7),
                                     xytext=(qy_m - 0.15, max(y) * 0.7),
                                     fontsize=10, color='black', alpha=0.7,
                                     arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        self.canvases[4].ax.axvline(qy_p, color='black', linestyle='--', alpha=0.5)
        self.canvases[4].ax.annotate(f'Qy+s = {qy_p:.3f}',
                                     xy=(qy_p, max(y) * 0.5),
                                     xytext=(qy_p + 0.05, max(y) * 0.5),
                                     fontsize=10, color='black', alpha=0.7,
                                     arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        self.canvases[4].ax.axvline(qs, color='black', linestyle='--', alpha=0.5)
        self.canvases[4].ax.annotate(f'Qs = {qs:.3f}',
                                     xy=(qs, max(y) * 0.3),
                                     xytext=(qs + 0.05, max(y) * 0.3),
                                     fontsize=10, color='black', alpha=0.7,
                                     arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        self.canvases[4].ax.grid(True, alpha=0.3)

        self.canvases[5].ax.clear()
        x = np.arange(1, len(self.data_manager.cy_array) + 1, 1)
        y = self.data_manager.cy_array
        self.canvases[5].ax.plot(x, y, 'ro-', linewidth=2)
        self.canvases[5].ax.set_title(self.data_manager.bpm_to_observe)
        self.canvases[5].ax.set_xlabel('Count')
        self.canvases[5].ax.set_ylabel('Chromaticity Cy')
        self.canvases[5].ax.grid(True, alpha=0.3)

        for canvas in self.canvases:
            canvas.restore_zoom()

            # Сохраняем исходный масштаб для каждого графика (если еще не сохранен)
        for canvas in self.canvases:
            if not canvas.initial_zoom_saved:
                canvas.save_original_zoom()

        for canvas in self.canvases:
            canvas.draw()

        self.x_tune_range_selector.update_visual()
        # self.s_tune_range_selector.update_visual()
        self.y_tune_range_selector.update_visual()

        self.data_manager.data["x"] = [np.nan]
        self.data_manager.data["y"] = [np.nan]

    def choose_bpm(self):
        button = self.sender()
        bpm = button.text()
        self.data_manager.bpm_to_observe = bpm
        self.data_manager.initialize_bpm_pv(bpm)

    def function2(self):
        pass

    def function3(self):
        pass

    def closeEvent(self, event):
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
