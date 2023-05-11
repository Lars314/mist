import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import os
import json
from scipy import stats
from scipy.optimize import curve_fit
import re

def prepare_LIDA_df(lab_path, wn_min, wn_max):
    df = pd.read_csv(lab_path, delim_whitespace=True,
                     names=["wavenumber", "absorbance"])
    df = df[(df['wavenumber'] > wn_min) &
            (df['wavenumber']< wn_max)].copy(deep=False)
    df['tau'] = df['absorbance']*np.log(10)
    return df

def prepare_Catania_df(lab_path, wn_min, wn_max):
    df = pd.read_csv(lab_path, engine="python", skiprows=20, header=None,
                     delim_whitespace=True, names=["wavenumber", "tau"])
    df = df[(df['wavenumber'] > wn_min) &
            (df['wavenumber']< wn_max)].copy(deep=False)
    return df

def evaluate_fit(ydata, yfit):
    """
    Uses the chi square to evaluate the goodness of fit. Returns the chi square,
    the reduced chi square, the p value, and the r2 statistic
    """

    # calculate chi^2
    chi2, p = stats.chisquare(ydata/np.sum(ydata), yfit/np.sum(yfit))
    redchi2 = chi2 / len(ydata)

    # residual sum of squares
    ss_res = np.sum((ydata - yfit) ** 2)
    # total sum of squares
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    # r-squared
    r2 = 1 - (ss_res / ss_tot)

    return chi2, redchi2, p, r2

def _atoi(text):
    """
    Implement natural sorting as from:
    https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    """
    return int(text) if text.isdigit() else text

def _natural_keys(text):
    """
    Implement natural sorting as from:
    https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    """
    return [ _atoi(c) for c in re.split(r'(\d+)', text) ]

def show_spectra(directory, wn_min, wn_max, do_vlines=False,
                 database="LIDA", t=None, n=None):
    """
    Plots the spectra at many temperatures
    -------- Parameters --------
    directory : the path to the folder with the spectra you want to see
    wn_min : the lower limit on wavenumber for plotting
    wn_max : the upper limit on wavenumber for plotting
    do_vlines : I was looking at the bending mode of CO2, this turns on or off
                vertical lines at the split peaks
    database : "LIDA" or "Catania", tells MIST how to read the data files
               depending on which database (and therefore format) they are from
    t : if using the LIDA database, you can set t to a specific temperature
        to just plot that spectrum
    n : if using the Catania database, you can set n to a specific name to just
        plot that spectrum
    """

    plt.rc('font', size=14)
    plt.rc('figure', dpi=100)
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 5)

    data = []
    max_temp = 1
    fnames = os.listdir(directory)
    fnames.sort(key=_natural_keys)
    for fname in fnames:
        if fname == ".ipynb_checkpoints":
            continue
        path = directory + "/" + fname

        if database == "LIDA" or database == "lida":
            if t != None:
                if float(fname[:-5]) != t:
                    continue
            this_df = prepare_LIDA_df(path, wn_min, wn_max)
            data.append({"name":fname[:-4], "df":this_df})
            if float(fname[:-5]) > max_temp:
                max_temp = float(fname[:-5])

        elif database == "Catania" or database == "catania":
            if n != None:
                if fname[:-4] != n:
                    continue
            this_df = prepare_Catania_df(path, wn_min, wn_max)
            data.append({"name":fname[:-4], "df":this_df})


    norm=plt.Normalize(0, max_temp)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "violet", "red"])

    if do_vlines:
        ax.axvline(661.25, color="xkcd:grey", linestyle="--")
        ax.axvline(654.59, color="xkcd:grey", linestyle="--")

    """    if database == "LIDA" or database == "lida":
        for spec in data:
            ax.plot(spec['df']['wavenumber'], spec['df']['tau'],
                    label=str(spec['temp'])+"K")
    else:"""
    for spec in data:
        ax.plot(spec['df']['wavenumber'], spec['df']['tau'],
                label=spec['name'])

    ax.set_xlim(wn_max, wn_min)
    ax.set_xlabel("wavenumber (1/cm)")
    ax.set_ylabel("Optical Depth")
    ax.invert_yaxis()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True,
              ncol=5, framealpha=0, fontsize=12)
    plt.close()
    return fig

def column_density(spec, ls=1):
    """
    Calculate column density in the optically thin limit
    """
    a = spec['weight']
    ydata = [a*t for t in spec['df']['flattened_tau']]
    xdata = list(spec['df']['wavenumber'])

    # numpy integration needs increasing functions
    if xdata[1] < xdata[0]:
        xdata = np.flip(xdata)
        ydata = np.flip(ydata)

    return ls*np.trapz(y=ydata, x=xdata)

class Fitter:
    def __init__(self, spec_path, wn_min, wn_max):
        self.path = spec_path
        self.wn_min = wn_min
        self.wn_max = wn_max
        
        # set up the spectra
        obs, lab, model_name = self._read_components()
        self.obs = obs
        self.lab = lab
        self.model_name = model_name
        
        # make combined spectrum
        combined = self._add_curves()
        self.fit_curve = combined

    def _prepare_LIDA_df(self, lab_path):
        df = pd.read_csv(lab_path, delim_whitespace=True,
                         names=["wavenumber", "absorbance"])
        df = df[(df['wavenumber'] > self.wn_min) &
                (df['wavenumber']< self.wn_max)].copy(deep=False)
        df['tau'] = df['absorbance']*np.log(10)
        return df

    def _prepare_Catania_df(self, lab_path):
        df = pd.read_csv(lab_path, engine="python", skiprows=20, header=None,
                         delim_whitespace=True, names=["wavenumber", "tau"])
        df = df[(df['wavenumber'] > self.wn_min) &
                (df['wavenumber']< self.wn_max)].copy(deep=False)
        return df

    def _flatten(self, curve, lower=620, upper=677, n_lower=100, n_upper=100,
                debug=False, warnings=False):
        """
        Flattens a curve to make it easier for fitting.
        """
        if debug:
            print("Flattening with:\nlower={0}\nupper={1}\nn_lower={2}"+
                  "\nn_upper={3}\n".format(lower, upper, n_lower, n_upper))

        curve = curve.reset_index()
        # check that the region we fit with is ok
        if warnings:
            if curve.loc[n_lower]['wavenumber'] > lower:
                print("Warning, lower limit is exceeded with of n={0}".format(n_lower))
            if curve.loc[len(curve)-1-n_upper]['wavenumber'] < upper:
                print("Warning, upper limit is exceeded with of n={0}".format(n_upper))

        if debug:
            print("curve region is between"+
                  "{0:.3f} and {1:.3f}".format(curve.loc[n_lower]['wavenumber'],
                                               curve.loc[len(curve)-1-n_upper]['wavenumber']))

        y1 = np.mean(curve[:n_lower]['tau'])
        y2 = np.mean(curve[-1*n_upper:]['tau'])
        x1 = np.mean(curve[:n_lower]['wavenumber'])
        x2 = np.mean(curve[-1*n_upper:]['wavenumber'])

        if debug:
            print("y1={0}\ny2={1}\nx1={2}\nx2={3}\n".format(y1, y2, x1, x2))

        # compute slope
        m = (y2-y1)/(x2-x1)

        if debug:
            print("m={0}\n".format(m))

        # create linear function in point-slope form
        X = curve['wavenumber']
        Y = (m*(X-x1) + y1)

        flattened = [this_y-this_Y for this_y, this_Y in zip(curve['tau'], Y)]
        # normalize to the same peak as before
        # actually do not do this as it changes the profile!
        #flattened = (flattened/np.max(flattened))*np.max(curve['tau'])

        # subtract to get the baseline at 0
        fy1 = np.mean(flattened[:n_upper])
        fy2 = np.mean(flattened[-n_lower:])
        baseline = np.mean([fy1, fy2])

        if debug:
            print("fy1={0}\nfy2={1}\nbaseline={2}\n".format(fy1, fy2, baseline))

        flattened_subtracted = flattened - baseline

        if debug:
            return flattened_subtracted, flattened
        else:
            return flattened_subtracted

    def _make_model_name(self, lab):
        model_name = ""
        for spectrum in lab:
            if spectrum['weight'] == 0:
                continue
            else:
                if model_name != "":
                    model_name += "+"
                model_name += "{0:4f}".format(spectrum['weight']) + "*({0})".format(spectrum['name'])
        return model_name

    def _read_components(self):
        """
        Reads a json file and extracts the components for this fit
        """
        # load the json
        with open(self.path) as f:
            data = json.load(f)
            f.close()
    
        # assign the observed and lab components
        obs = data['observed']
        lab = data['lab']

        # format the observed data into a dataframe
        if obs['name'] == "JWST Data Yang et al.":
            obs_df = pd.read_csv(obs["path"], sep=" \s+", engine='python')
            obs_df['wavenumber'] = (10**4)/(obs_df['wavelength(um)']+0.02)
            obs["df"] = obs_df
            
        elif obs['name'] == "Elias 29":
            # the data file from Sergio
            df = pd.read_csv(obs["path"], delim_whitespace=True,
                                 names=["lambda (um)", "Flux (Jy)",
                                        "Sigma (Jy)", "AOT ident."], skiprows=6)
            # convert um to wavenumbers
            df['wavenumber'] = (10**4)/(df['lambda (um)']+0.02)
            
            # apply limits
            obs_df = df[(df['wavenumber'] > self.wn_min) & \
                        (df['wavenumber'] < self.wn_max)].copy(deep=True)

            # this file is in flux units, we need to get optical depth.
            # to do that we use the formula tau = -ln(fo/fc) where tau is the
            # optical depth, fo is the observed flux, and fc is the continuum
            
            # read the continuum, taken from LIDA
            cont = pd.read_csv("./data/all_SED/2.txt", delim_whitespace=True,
                               names=['wavenumber (um)', 'Flux (Jy)'])
            # we need Fo and Fc to match in wavenumber space, so interpolate
            interp_cont = np.interp(x=obs_df['lambda (um)'],
                                    xp=cont['wavenumber (um)'],
                                    fp=cont['Flux (Jy)'])
            # now we can calculate tau
            obs_df['tau'] = [-np.log(fo/fc) for fo, fc in \
                             zip(obs_df['Flux (Jy)'], interp_cont)]
            # also calculate its error
            obs_df['error_tau'] = [sigf/f for f, sigf in \
                                   zip(obs_df['Flux (Jy)'],
                                       obs_df['Sigma (Jy)'])]
            obs["df"] = obs_df

        else:
            print("Observational Data {0} not recognized.".format(obs['name']))

        # format the lab data
        nonzero_components = []
        for spectrum in lab:
            # we only care about non-zero components
            if spectrum['weight'] != 0:
                # get the lab data into dataframes with the right wavenumber limits
                if spectrum['database'] == "LIDA":
                    spectrum['df'] = prepare_LIDA_df(spectrum['path'],
                                                     self.wn_min, self.wn_max)
                    spectrum['df']['flattened_tau']= self._flatten(spectrum['df'],
                                                             n_upper=spectrum['n_upper'],
                                                             n_lower=spectrum['n_lower'])
                elif spectrum['database'] == "Catania":
                    spectrum['df'] = prepare_Catania_df(spectrum['path'],
                                                       self.wn_min, self.wn_max)
                    # flatten the lab data, removing any non-zero baseline
                    #spectrum['df']['flattened_tau'] = spectrum['df']['tau']
                    spectrum['df']['flattened_tau']= self._flatten(spectrum['df'],
                                                             n_upper=spectrum['n_upper'],
                                                             n_lower=spectrum['n_lower'])
                    #test = spectrum['df']

                nonzero_components.append(spectrum)
            else:
                continue

        # name the model based on which components have non-zero weights
        model_name = self._make_model_name(lab)
        #print("The model name is: \n" + model_name)

        return obs, nonzero_components, model_name
    
    def _one_component_model(self, wavenumbers, w0):
        """
        Creates a one component model in a way that can be used with scipy's
        curve_fit. The components must be in "spectra.json", and the weights there
        will be overridden and computed based on the fit
        """
        self.lab[0]['weight'] = w0

        self.model = self._add_curves()

        return self.model['tau']
    
    def _two_component_model(self, wavenumbers, w0, w1):
        """
        Creates a two component model in a way that can be used with scipy's
        curve_fit. The components must be in "spectra.json", and the weights there
        will be overridden and computed based on the fit
        """
        self.lab[0]['weight'] = w0
        self.lab[1]['weight'] = w1

        self.model = self._add_curves()

        return self.model['tau']

    def _three_component_model(self, wavenumbers, w0, w1, w2):
        """
        Creates a three component model in a way that can be used with scipy's
        curve_fit. The components must be in "spectra.json", and the weights there
        will be overridden and computed based on the fit
        """
        self.lab[0]['weight'] = w0
        self.lab[1]['weight'] = w1
        self.lab[2]['weight'] = w2

        self.model = self._add_curves()

        return self.model['tau']

    def _four_component_model(self, wavenumbers, w0, w1, w2, w3):
        """
        Creates a four component model in a way that can be used with scipy's
        curve_fit. The components must be in "spectra.json", and the weights there
        will be overridden and computed based on the fit
        """
        self.lab[0]['weight'] = w0
        self.lab[1]['weight'] = w1
        self.lab[2]['weight'] = w2
        self.lab[3]['weight'] = w3

        self.model = self._add_curves()

        return self.model['tau']

    def _five_component_model(self, wavenumbers, w0, w1, w2, w3, w4):
        """
        Creates a five component model in a way that can be used with scipy's
        curve_fit. The components must be in "spectra.json", and the weights there
        will be overridden and computed based on the fit
        """
        self.lab[0]['weight'] = w0
        self.lab[1]['weight'] = w1
        self.lab[2]['weight'] = w2
        self.lab[3]['weight'] = w3
        self.lab[4]['weight'] = w4

        self.model = self._add_curves()

        return self.model['tau']

    def _six_component_model(self, wavenumbers, w0, w1, w2, w3, w4, w5):
        """
        Creates a six component model in a way that can be used with scipy's
        curve_fit. The components must be in "spectra.json", and the weights there
        will be overridden and computed based on the fit
        """
        self.lab[0]['weight'] = w0
        self.lab[1]['weight'] = w1
        self.lab[2]['weight'] = w2
        self.lab[3]['weight'] = w3
        self.lab[4]['weight'] = w4
        self.lab[5]['weight'] = w5

        self.model = self._add_curves()

        return self.model['tau']
    
    def _seven_component_model(self, wavenumbers, w0, w1, w2, w3, w4, w5, w6):
        """
        Creates a seven component model in a way that can be used with scipy's
        curve_fit. The components must be in "spectra.json", and the weights there
        will be overridden and computed based on the fit
        """
        self.lab[0]['weight'] = w0
        self.lab[1]['weight'] = w1
        self.lab[2]['weight'] = w2
        self.lab[3]['weight'] = w3
        self.lab[4]['weight'] = w4
        self.lab[5]['weight'] = w5
        self.lab[6]['weight'] = w6

        self.model = self._add_curves()

        return self.model['tau']
    
    def _eight_component_model(self, wavenumbers, w0, w1, w2, w3, w4, w5, w6, w7):
        """
        Creates an eight component model in a way that can be used with scipy's
        curve_fit
        """
        self.lab[0]['weight'] = w0
        self.lab[1]['weight'] = w1
        self.lab[2]['weight'] = w2
        self.lab[3]['weight'] = w3
        self.lab[4]['weight'] = w4
        self.lab[5]['weight'] = w5
        self.lab[6]['weight'] = w6
        self.lab[7]['weight'] = w7

        self.model = self._add_curves()

        return self.model['tau']
    
    def _nine_component_model(self, wavenumbers, w0, w1, w2, w3, w4, w5, w6, w7, w8):
        """
        Creates a nine component model in a way that can be used with scipy's
        curve_fit
        """
        self.lab[0]['weight'] = w0
        self.lab[1]['weight'] = w1
        self.lab[2]['weight'] = w2
        self.lab[3]['weight'] = w3
        self.lab[4]['weight'] = w4
        self.lab[5]['weight'] = w5
        self.lab[6]['weight'] = w6
        self.lab[7]['weight'] = w7
        self.lab[8]['weight'] = w8

        self.model = self._add_curves()

        return self.model['tau']
    
    def _ten_component_model(self, wavenumbers, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9):
        """
        Creates a ten component model in a way that can be used with scipy's
        curve_fit
        """
        self.lab[0]['weight'] = w0
        self.lab[1]['weight'] = w1
        self.lab[2]['weight'] = w2
        self.lab[3]['weight'] = w3
        self.lab[4]['weight'] = w4
        self.lab[5]['weight'] = w5
        self.lab[6]['weight'] = w6
        self.lab[7]['weight'] = w7
        self.lab[8]['weight'] = w8
        self.lab[9]['weight'] = w9

        self.model = self._add_curves()

        return self.model['tau']

    def _add_curves(self):
        """
        Returns a linear combination of two curves
        """
        wavenumbers = self.obs['df']['wavenumber']
        combined = {"wavenumber":wavenumbers, "tau":[]}
        wavenumbers = list(wavenumbers)
        
        for spectrum in self.lab:

            # the stuff to combine
            this_A = spectrum['weight']
            this_wavenumber = list(spectrum['df']['wavenumber'])
            this_tau = list(this_A*spectrum['df']['flattened_tau'])

            # interpolate needs increasing functions
            if this_wavenumber[1] < this_wavenumber[0]:
                this_wavenumber = np.flip(this_wavenumber)
                this_tau = np.flip(this_tau)


            interp_tau = np.interp(x=combined['wavenumber'], xp=this_wavenumber, fp=this_tau)

            # if this is the first curve added, just make it combined
            if not any(combined['tau']):
                combined['tau'] = interp_tau
            # otherwise, we need to add them
            else:
                combined['tau'] = [y1 + y2 for y1, y2 in zip(combined['tau'], interp_tau)]
        return combined

    def plot_spectra(self, save_model=False, do_vlines=False, do_eval=True):
        """
        Makes a plot
        """
        plt.rc('font', size=14)
        plt.rc('figure', dpi=300)
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(10, 5)

        if do_vlines:
            ax.axvline(661.25, color="xkcd:grey", linestyle="--")
            ax.axvline(654.59, color="xkcd:grey", linestyle="--")

        ax.errorbar(self.obs['df']['wavenumber'], self.obs['df']['tau'],
                    yerr=self.obs['df']['error_tau'], label=self.obs['name'],
                    linestyle='-', marker='o', markersize=2)

        # loop over the components
        for spectrum in self.lab:
            weight = spectrum['weight']
            if weight == 0:
                continue
            else:
                ax.plot(spectrum['df']['wavenumber'],
                        weight*spectrum['df']['flattened_tau'],
                        label="{0:.4f}*({1})".format(weight,
                                                     spectrum['name']), alpha=0.75)

        # plot the combined curve
        ax.plot(self.fit_curve['wavenumber'], self.fit_curve['tau'], label="Model", color="xkcd:black", linewidth=3, alpha=1)

        #ax.set_xlim(self.wn_min, self.wn_max)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel("wavenumber (1/cm)")
        ax.set_ylabel("Optical Depth");
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True,
                  ncol=3, framealpha=0, fontsize=12)

        if do_eval:
            ydata = self.obs['df']['tau']
            chi2, redchi2, p, r2 = evaluate_fit(ydata, self.fit_curve['tau'])
            ax.set_title(r"$\chi^2=$"+"{0:.4f}".format(chi2) + "   " +
                         " p-value={0:.2f}".format(p) + "   " +
                         r"$R^2=$" + "{0:.4f}".format(r2))

        if save_model:
            plt.savefig("./models/{0}.jpg".format(self.model_name.replace(" ", "_")), bbox_inches="tight")
        plt.close()
        return fig

    def do_fit(self, bounds=(0, 100)):
        p0 = []
        for spectrum in self.lab:
            p0.append(spectrum['weight'])

        xdata = self.obs['df']['wavenumber']
        ydata = self.obs['df']['tau']

        n = len(p0)
        if n == 10:
            popt, pcov = curve_fit(self._ten_component_model, xdata, ydata,
                                   bounds=(0, 100), p0=p0)
            model = self._ten_component_model(xdata, popt[0], popt[1], popt[2],
                                        popt[3], popt[4], popt[5], popt[6],
                                        popt[7], popt[8], popt[9])
        elif n == 9:
            popt, pcov = curve_fit(self._nine_component_model, xdata, ydata,
                                   bounds=(0, 100), p0=p0)
            model = self._nine_component_model(xdata, popt[0], popt[1], popt[2],
                                        popt[3], popt[4], popt[5], popt[6],
                                        popt[7], popt[8])
        elif n == 8:
            popt, pcov = curve_fit(self._eight_component_model, xdata, ydata,
                                   bounds=(0, 100), p0=p0)
            model = self._eight_component_model(xdata, popt[0], popt[1], popt[2],
                                        popt[3], popt[4], popt[5], popt[6],
                                        popt[7])
        elif n == 7:
            popt, pcov = curve_fit(self._seven_component_model, xdata, ydata,
                                   bounds=(0, 100), p0=p0)
            model = self._seven_component_model(xdata, popt[0], popt[1], popt[2],
                                        popt[3], popt[4], popt[5], popt[6])
        elif n == 6:
            popt, pcov = curve_fit(self._six_component_model, xdata, ydata,
                                   bounds=(0, 100), p0=p0)
            model = self._six_component_model(xdata, popt[0], popt[1], popt[2],
                                        popt[3], popt[4], popt[5])
        elif n == 5:
            popt, pcov = curve_fit(self._five_component_model, xdata, ydata,
                                   bounds=(0, 100), p0=p0)
            model = self._five_component_model(xdata, popt[0], popt[1],
                                         popt[2], popt[3], popt[4])
        elif n == 4:
            popt, pcov = curve_fit(self._four_component_model, xdata, ydata,
                                   bounds=(0, 100), p0=p0)
            model = self._four_component_model(xdata, popt[0], popt[1],
                                         popt[2], popt[3])
        elif n == 3:
            popt, pcov = curve_fit(self._three_component_model, xdata, ydata,
                                   bounds=(0, 100), p0=p0)
            model = self._three_component_model(xdata, popt[0], popt[1], popt[2])
        elif n == 2:
            popt, pcov = curve_fit(self._two_component_model, xdata, ydata,
                                   bounds=(0, 100), p0=p0)
            model = self._two_component_model(xdata, popt[0], popt[1])
        elif n == 1:
            popt, pcov = curve_fit(self._one_component_model, xdata, ydata,
                                   bounds=(0, 100), p0=p0)
            model = self._one_component_model(xdata, popt[0])
        else:
            print("{0} component model is not supported".format(n))

        # update the model with the new weights
        for i in range(0, len(self.lab)):
            self.lab[i]['weight'] = popt[i]

        combined = self._add_curves()

        # refresh the model name to include the new weights
        model_name = self._make_model_name(self.lab)

        # plot the spectra and fit
        #fitter.plot_spectra(obs, lab, combined, model_name, save_model=True)
        self.fit_curve = combined
        self.model_name = model_name
        self.p0 = p0
        #return lab, combined, model_name, p0
        
    def analyze_components(self):
        """
        Compute the column densities of the different componenets
        """

        self.col_d = None
