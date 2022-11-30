#This code contains useful tools for personal using of PyCBC
import numpy as np
import matplotlib.pyplot as plt


#============================================================================#
#                                                                            #
#                                Useful Constants                            #
#                                                                            #
#============================================================================#


speed_of_light_mks = 299792458
r_trans = 1.4766250385
p_mks_to_mev = 1.60218e32
rho_mks_to_mev = 1.782661907e15
p_geo_to_mks = 8.262346242653866e-45
rho_geo_to_mks = 7.425826474101849e-28
p_cactus_to_cgs = 1.80123683248503e-39
rho_cactus_to_cgs = 1.61887093132742e-18

my_beauty_colors = [(0.9756082618370889, 0.8819298114153367, 0.15511474788168034), 
            (0.8816686822876404, 0.16723558098223046, 0.8232440687726762),
            (0.09474917006457495, 0.7640816932916719, 0.7557809243480891),
            (0.9869622156803993, 0.19773950935450402, 0.11477842930491322),
            (0.0013876747101176923, 0.09891733102233669, 0.21359016748958548), 
            (0.005058894123937008, 0.19158725163173618, 0.8274597939134388),
            (0.14066189835392162, 0.9108866866830906, 0.43832418793816164),
            (0.4795075899295418, 0.09701374232611071, 0.867789302976063), 
            (0.101188878202501, 0.4761126547943416, 0.8337685909406162),
            (0.732657217071116, 0.028265149660644195, 0.44243490354748116),
            (0.47075839622563465, 0.1865018861754263, 0.27292819189751183), 
            (0.012832705028862468, 0.6240926466800075, 0.07013874925574748)] 


class my_color():
    def __init__(self, cmap, vmin, vmax):
        """ get scallable colors
        """
        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        self.colors = mcm.ScalarMappable(norm=norm, cmap=cmap)
        
    def get_color(self, x):
        return self.colors.to_rgba(x)
#============================================================================#
#                                                                            #
#                   Calculate & Show Sample Properties                       #
#                                                                            #
#============================================================================#


def hpd_interval(post_sample, conf_interval=90, prior_sample=None, \
                 param_min=None, param_max=None, prior_method="div_kde", \
                 bw_method='scott', bins="rice", root_frac=(0.5, 0.5), \
                 pdf_guess_fac=1., pdf_guess=None, xtol=1e-7, ftol=1e-10, \
                 return_sample=False, return_size=2000, show_kde=False, \
                 debug=False):
    """Calculate Highest Posterior Density Interval for given posterior sample

    Parameters
    ----------
    post_sample: one dimensional array like 
        posterior sample to calculate HPD.
    conf_interval: float
        credible interval of HPD.
    prior_sample: one dimensional array like, optional
        prior sample, if used, will calculate HPD of posterior/prior.
    param_min: float, optional
         lower bound to normalize KDE, if not given, simply min of all sample.
    param_max: float, optional
         upper bound to normalize KDE, if not given, simply max of all sample.
    prior_method: str, optional
        determine which method to divide post by prior, 'div_kde' or 'div_bin'
    bw_method: str, scalar or callable, optional
        see scipy.stats.gaussian_kde
    bins: str, int or 1d array like:
        see numpy.histogram
    pdf_guess_fac: float, optional
        a avoid the default value of pdf_guess is the highest value of pdf
    root_frac: tuple of two float, optional
        search first root in (param_min, param_min+root_frac[0]*(param_max-
        param_min)), and second root in (param_min+root_frac[1]*(param_max-
        param_min), param_max)
    pdf_guess: float, optional
        if given, will not find intersection, use pdf_gusess to get interval
    xtol: float
        see scipy.optimize.fmin
    ftol: float
        see scipy.optimize.fmin
    return_sample: bool, optional
        if given return kde samples of return_size
    show_kde: bool, optional
        if set to true, plot kde figure.
    debug: bool, optional
        if set to true, print important intermediary results

    Returns
    -------
    Sample of two dimension with 'x' the first axis and 'y' the second
        if return_sample is set to True
    HPD interval:tuple
        (HPD lower, PPD median, HPD upper)
    """
    from scipy.integrate import quad
    from scipy.optimize import fmin, ridder
    from scipy.stats import gaussian_kde
    from decimal import Decimal

    comp = prior_sample if hasattr(prior_sample, '__iter__') else post_sample
    if param_min==None:
        param_min = np.min(np.hstack((post_sample, comp)))
    if param_max==None:
        param_max = np.max(np.hstack((post_sample, comp)))

    def find_intersec(intersec, func):
        def f(x):
            pdf = func(x)
            return pdf if pdf>intersec else 0
        prob = quad(f, param_min, param_max)[0]
        if debug:
            print("Wanted prob: {}".format(conf_interval/100.))
            print("\tGet: {}, pdf value: {}".format(prob, intersec[0]))
        return (prob-conf_interval/100.)**2

    def find_root(func, intersec, lower_guess, upper_guess):
        return ridder(lambda x: func(x)-intersec, lower_guess, upper_guess)

    def post_devide_prio(post_sample, prior_sample):
        sorted_pr = np.array(sorted(prior_sample))
        sorted_po = np.array(sorted(post_sample))
        sorted_pr = sorted_pr[(sorted_pr<=param_max)*(sorted_pr>=param_min)]
        sorted_po = sorted_po[(sorted_po<=param_max)*(sorted_po>=param_min)]
        prio_hist, edges = np.histogram(sorted_pr, bins=bins)
        post_hist, _ = np.histogram(sorted_po, bins=edges)
        probs = [(float(po)/pr if pr!=0. else float(po)) for po, pr in zip(post_hist, prio_hist)]
        if debug:
            print("Divided into {} bins.".format(len(probs)))
            print("Edges: {}".format(edges))
            print("Divided result: {}".format(probs))
        normed_prob = np.array(probs)/sum(probs)
        wanted_num = len(prior_sample)
        choices = np.random.choice(edges[:-1], size=wanted_num, p=normed_prob)
        ret = []
        for ch in choices:
            uniform_up = edges[np.where(edges==ch)[0][0]+1]
            ret.append(np.random.uniform(ch, uniform_up))
        return ret

    post_kde = gaussian_kde(post_sample, bw_method=bw_method)
    if not hasattr(prior_sample, '__iter__'):
        eval_sample = post_sample
        normed_fac = quad(post_kde, param_min, param_max)[0]
        normed_post = lambda x:post_kde(x)/normed_fac
        eval_kde = normed_post
    else:
        if prior_method=="div_kde":
            eval_sample = post_sample
            prior_kde = gaussian_kde(prior_sample, bw_method=bw_method)
            likelihood_kde = lambda x:post_kde(x)/prior_kde(x)
        else:
            eval_sample = post_devide_prio(post_sample, prior_sample)
            likelihood_kde = gaussian_kde(eval_sample, bw_method=bw_method)
        normed_fac = quad(likelihood_kde, param_min, param_max)[0]
        eval_kde = lambda x: likelihood_kde(x)/normed_fac

    if return_sample:
        x = np.linspace(param_min, param_max, return_size)
        y = eval_kde(x)
        return np.vstack((x,y))

    if pdf_guess==None:
        pdf_guess = eval_kde(np.median(eval_sample))*pdf_guess_fac
        intersec = fmin(find_intersec, x0=pdf_guess, xtol=xtol, ftol=ftol, \
                                                         args=(eval_kde, ))[0]
    else:
        intersec = pdf_guess
    first_root_upp = param_min+root_frac[0]*(param_max-param_min)
    second_root_low = param_min+root_frac[1]*(param_max-param_min)
    get_median = lambda x: quad(eval_kde, param_min, x)[0]-0.5

    if debug:
        diff_1 = eval_kde(param_min)[0]-intersec
        diff_2 = eval_kde(first_root_upp)[0]-intersec
        diff_3 = eval_kde(second_root_low)[0]-intersec
        diff_4 = eval_kde(param_max)[0]-intersec
        diff_5 = get_median(param_min)
        diff_6 = get_median(param_max)
        print("Intersection: {:.4E}".format(Decimal(str(intersec))))
        print("Search 1st root in:({},{})".format(param_min, first_root_upp))
        print("\tFunc_value-intersection:({},{})".format(diff_1, diff_2))
        print("Search 2nd root in:({},{})".format(second_root_low, param_max))
        print("\tFunc_value-intersection:({},{})".format(diff_3, diff_4))
        print("Search median in:({},{})".format(param_min, param_max))
        print("\tIntegrated probability-0.5:({},{})".format(diff_5, diff_6))

    hpd_lower = find_root(eval_kde, intersec, param_min, first_root_upp)
    hpd_upper = find_root(eval_kde, intersec, second_root_low, param_max)
    hpd_median = ridder(get_median, param_min, param_max)

    if show_kde:
        x = np.linspace(param_min, param_max, 1000)
        y = eval_kde(x)
        pos = post_sample
        prs = prior_sample
        es = eval_sample
        plt.plot(x, y, lw=3, label="HPD KDE")
        if not hasattr(prior_sample, '__iter__'):
            l = "post bins"
            plt.hist(pos, bins=bins, alpha=0.7, density=True, label=l)
        else:
            plt.plot(x, post_kde(x), lw=2, ls='--', c='k', label="post KDE")
            if prior_method=="div_kde":
                l = "prior KDE"
                plt.plot(x, prior_kde(x), lw=2, ls='-.', c='k', label=l)
                l = "prior bins"
                plt.hist(prs, bins=bins, alpha=0.7, density=True,label=l)
                l = "post bins"
                plt.hist(pos, bins=bins, alpha=0.7, density=True, label=l)
            else:
                l = "resampled bins"
                plt.hist(es, bins=bins, alpha=0.7, density=True,label=l)
        lb = r"{}% HPD interval".format(conf_interval)
        plt.vlines((hpd_lower, hpd_upper), 0, max(y)*1.2, colors='g',label=lb)
        plt.vlines(hpd_median, 0, max(y)*1.2, colors='m', label="hpd_median")
        lb = "Intersection: {:.2E}".format(Decimal(str(intersec)))
        plt.hlines(intersec, param_min, param_max, colors='r',label=lb)
        plt.xlim(param_min, param_max)
        plus = hpd_upper-hpd_median
        minus = hpd_lower-hpd_median
        formats = r"HPD interval: ${{{:.2f}}}^{{{:+.2f}}}_{{{:+.2f}}}$"
        plt.title(formats.format(hpd_median, plus, minus))
        plt.legend()
        plt.show()
    return hpd_lower, hpd_median, hpd_upper


def cal_sample_property(sample, conf_interval=90., method="median", diff=True, round_level=2):
    """Return  lower confidence bound, mean(or median), upper confidence bound. If diff
    is set to True, return (middle-lower, middle, upper-middle)
    """

    conf_lower = np.percentile(sample, (100-conf_interval)/2.)
    conf_upper = np.percentile(sample, (100+conf_interval)/2.)
    out = np.median(sample) if method=="median" else np.mean(sample)
    if diff:
        conf_lower = out-conf_lower
        conf_upper = conf_upper-out
    return np.round([conf_lower, out, conf_upper], round_level)


def interval_string(arr, precision=2, conf_interval=90.0):
    if precision<=0:
        format_string = r"${:.0f}_{{-{:.0f}}}^{{+{:.0f}}}$"
    else:
        format_string = r"${:."+str(precision)+r"f}_{{-{:."+str(precision)+r"f}}}^{{+{:."+str(precision)+r"f}}}$"
    stats = np.array(cal_sample_property(arr, conf_interval=conf_interval, diff=True, method="median", round_level=5))
    return format_string.format(np.round(stats[1], precision), np.round(stats[0], precision), np.round(stats[2], precision))


def cal_pdf_property(x, pdf, conf_interval=90., hpd=False, diff=True, round_level=2, debug=False):
    """Return lower (HPD)confidence bound, median, upper (HPD)confidence bound of a 
    function. x and pdf must have the same size. x do not need to be equally separated and 
    pdf do not need to be normalized. More pdf samples give more accurate properties.
    """
    from scipy.optimize import fmin

    def find_intersec(intersec, dx, pdf):
        idx = pdf>intersec
        prob = sum(pdf[idx]*dx[idx])
        if debug:
            print("Wanted prob: {}".format(conf_interval/100.))
            print("\tGet: {}, pdf value: {}".format(prob, intersec[0]))
        return (prob-conf_interval/100.)**2

    pdf = (pdf[:-1]+pdf[1:])/2.
    dx = x[1:]-x[:-1]
    pdf /= sum(dx*pdf)
    cumsum_pdf = np.insert(np.cumsum(pdf*dx), 0, 0.0)
    if debug:
        print("normalized: {}".format(cumsum_pdf[-1]))
    if hpd:
        intersec = fmin(find_intersec, x0=max(pdf)/2., args=(dx, pdf), disp=False)
        larger_than_intersec_idxes = np.where(pdf>intersec)[0]
        lower_idx = larger_than_intersec_idxes[0]
        upper_idx = larger_than_intersec_idxes[-1]
    else:
        conf_l = (100.-conf_interval)/2./100.
        conf_u = (100.+conf_interval)/2./100.
        lower_idx = np.where(cumsum_pdf>=conf_l)[0][0]
        upper_idx = np.where(cumsum_pdf>=conf_u)[0][0]
    median_idx = np.where(cumsum_pdf>=0.5)[0][0]
    if debug:
        print("intersection:{}".format(intersec))
        print("idx_lower:{},idx_median:{},idx_upper:{}".format(lower_idx,median_idx,upper_idx))
        print("lower:{},median:{},upper:{}".format(cumsum_pdf[lower_idx],cumsum_pdf[median_idx],cumsum_pdf[upper_idx]))
    lower, median, upper = x[lower_idx], x[median_idx], x[upper_idx]
    if diff:
        lower = median-lower
        upper = upper-median
    return np.round([lower, median, upper], round_level)


def cal_acl(sample, nlags=128, window=5, show_plot=False, verbose=False):
    """Calculate acl.

    Parameters
    ----------
    sample: 1D array like
        Sample to calculate acl.
    nlags: int, optional
        Time lag that should be used to calculate acf.
    window: int
        When index>window*(1+2*sum(auto_cor, from 0 to index)), stop 
        accumulating acl.
    show_plot: bool, optional
        Show plot.
    verbose: bool, optional
        Print important intermediate results.
    """

    import math
    from statsmodels.tsa.stattools import acf

    auto_corf = acf(sample, nlags=nlags)
    N = len(sample)
    acl = -1
    success_state = False
    for M, ac in enumerate(auto_corf):
        acl += 2*ac
        if M>window*acl:
            success_state = True
            break
    if success_state:
        acl = int(math.ceil(acl))
    else:
        acl = np.inf
        print("Infinity acl, try increase nlags.")
    if verbose:
        print("acl<<window<<len_sample: {}<<{}<<{}".format(acl, M, N))
        var = 2.*(2*M+1)*acl**2/N
        print("var(acl):{}, sqrt(var)/acl: {}".format(var, np.sqrt(var)/acl))

    if show_plot:
        lac = nlags+1 #Length of autocorrelation coefficient
        x = [i for i in range(lac)]
        cr = auto_corf[M]
        plt.vlines(M, cr, 1, colors='b', lw=1, linestyle=':', label="window")
        plt.hlines(cr, 0, lac, colors='b', lw=1, linestyle='--', \
                   label="stop accumulating acf")
        plt.scatter(x, auto_corf, color='g', s=3, label="acf")
        plt.ylabel("acf")
        plt.xlabel("t_lag")
        plt.legend()
        plt.show()
    return acl


def plot_log_pm(arr, base=10):
    """Show log array and keep the sign
    """
    revised_arr = np.array([np.sign(it)*np.log(np.abs(it))/np.log(base) for it in arr])
    plt.scatter(np.arange(len(arr))+1, revised_arr)
    plt.show()


def plot_single(sample, true_value=None, percentiles=[5., 50., 95.], bins='auto', \
                xlabel='x', xscale='linear', xmin=None, xmax=None, filename=None):
    """Show single sample properties.

    Parameters
    ----------
    sample: 1D array like
        Sample to show properties
    percentiles: list of float
        Percentiles to be shown.
    bins: see numpy.histogram
    filename: str
        save to path 'filename' if given, show it directly it is default value
    """

    from scipy.stats import gaussian_kde

    mmin = min(sample)
    mmax = max(sample)
    x = np.linspace(mmin, mmax, 2000)
    kde = gaussian_kde(sample, bw_method="scott")
    y = kde(x)
    perc = [np.percentile(sample, p) for p in percentiles]
    y_max = max(y)
    fac = 1.2
    if xmin==None:
        xmin = mmin*fac if np.sign(mmin)==-1 else mmin/fac
    if xmax==None:
        xmax = mmax/fac if np.sign(mmax)==-1 else mmax*fac
    xbound = (xmin, xmax)
    ybound = (0, y_max*1.3)
    plt.xlim(xbound)
    plt.ylim(ybound)
    plt.plot(x, y, c='y', label="kde", lw=2)
    if true_value!=None:
        plt.vlines(true_value, 0, y_max*1.2, colors='r', lw=1, linestyle=':',\
                   label="true value")
    plt.vlines(perc, 0, y_max, colors='g', lw=1, linestyle='--')
    if xscale=='log':
        _, bins = np.histogram(np.log10(sample[sample!=0.]), bins=bins)
        plt.hist(sample, bins=10**bins, density=True, label="hist", alpha=0.7)
        xlabel = 'log10('+xlabel+')'
    else:
        plt.hist(sample, bins=bins, density=True, label="hist", alpha=0.7)
    plt.title(r"${{{:.2f}}}^{{{:+.2f}}}_{{{:+.2f}}}$".format(perc[1], perc[2]\
                   -perc[1], perc[0]-perc[1]))
    plt.ylabel("PDF")
    plt.xlabel(xlabel)
    plt.xscale(xscale)
    plt.legend()
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_density(sample1, sample2, true_values=None, xlabel=None, ylabel=None, \
                 m_pt=[5, 50, 95], c_pt=[68.3, 95.5], filename=None, \
                 lower_1=None, upper_1=None, lower_2=None, upper_2=None, **kargs):
    """Plot two variable density plot using PyCBC.

    Parameters
    ----------
    m_pt: list of float
        marginal_percentiles
    c_pt: list of float
        contour_percentiles
    filename: str
        save to path 'filename' if given, show it directly it is default value
    """

    from pycbc.results.scatter_histograms import create_multidim_plot

    if xlabel==None: xlabel='x'
    if ylabel==None: ylabel='y'
    names = [xlabel, ylabel]
    combined_sample = {xlabel: sample1, ylabel: sample2}
    if hasattr(true_values, '__iter__'):
        true_values = {xlabel: true_values[0], ylabel: true_values[1]}
    create_multidim_plot(names, combined_sample, show_colorbar=False, expected_parameters=\
                  true_values, plot_density=True, mins={xlabel:lower_1, ylabel:lower_2}, \
                  maxs={xlabel:upper_1, ylabel:upper_2}, density_cmap='BuPu', \
                  plot_scatter=False, marginal_percentiles=m_pt, \
                  contour_percentiles=c_pt, **kargs)
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_unique(sample, show_repeated=False, filename=None):
    """Show how unique data points in a sample vary with time.
    
    Parameters
    ----------
    sample: 1D array like
        If the data is two dimensional, it will be averaged along the 0-axis.
    show_repeated: bool, optional
        Whether to show the samples repeated.
    filename: str
        save to path 'filename' if given, show it directly it is default value
    """
    sample = np.array(sample)
    if len(sample.shape)==2:
        sample = np.mean(sample, axis=0)
    thinned = []
    repeated = [False]
    thinned.append(sample[0])
    for i, s in enumerate(sample[1:]):
        if s!=sample[i]:
            thinned.append(s)
            repeated.append(False)
        else:
            repeated.append(True)
    repeat_s = sample[np.array(repeated)]
    print("Length of sample:{}, unique sample points:{}, unique fraction:{}"\
          .format(len(sample), len(thinned), float(len(thinned))/len(sample)))
    x1 = [i+1 for i, r in enumerate(repeated) if not r]
    x2 = [i+1 for i, r in enumerate(repeated) if r]
    label = "not repeated sample points"
    plt.scatter(x1, thinned, color='r', s=3, label=label)
    plt.plot(x1, thinned, c='g', lw=0.5, ls=':')
    if show_repeated:
        label = "repeated sample points"
        plt.scatter(x2, repeat_s, color='y', s=1, label=label)
    plt.xlabel("step")
    plt.legend()
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)


#============================================================================#
#                                                                            #
#                            Show Sample Differences                         #
#                                                                            #
#============================================================================#


def cal_kldiv(sample1, sample2, base=2., bw_method='scott', bins="rice", \
              use_kde=False, cal_jsd=False):
    """Calculate Kullback-Leibler divergence: KL(sample1|sample2)

    Parameters
    ----------
    sample1: 1D array like
        sample1 to calculate
    sample2: 1D array like
        sample2 to calculate
    base: float, optional
        log base to calculate KL
    bw_method: bw_method
        see doc of scipy.stats.gaussian_kde
    bins: bins 
        see doc of numpy.histogram
    use_kde: bool, optional
        use KDE method to mimic sample behavior or just use histogram method
    cal_jsd: bool, optional
        set to true to calculate Jensen-Shannon divergence

    Returns
    -------
    divergence: float
        return JDS if cal_jsd else KLD
    """

    from scipy.stats import gaussian_kde
    from scipy.integrate import quad
    
    hist_ignore, bounds = np.histogram(np.append(sample1, sample2), bins=bins)
    if use_kde:
        lower_bds = bounds[:-1]
        upper_bds = bounds[1:]
        func1 = gaussian_kde(sample1, bw_method=bw_method)
        func2 = gaussian_kde(sample2, bw_method=bw_method)
        hist1 = [quad(func1, lb, ub)[0] for lb, ub in zip(lower_bds, upper_bds)]
        hist2 = [quad(func2, lb, ub)[0] for lb, ub in zip(lower_bds, upper_bds)]
    else:
        hist1, _ = np.histogram(sample1, bounds)
        hist2, _ = np.histogram(sample2, bounds)
    hist1 = np.array(hist1)/float(sum(hist1))
    hist2 = np.array(hist2)/float(sum(hist2))
    if cal_jsd:
        from scipy.spatial import distance
        js_div = (distance.jensenshannon(hist1, hist2, base=base))**2
    else:
        from scipy.special import rel_entr
        kl_div = sum(rel_entr(hist1[hist2>0], hist2[hist2>0]))/np.log(base)
    return js_div if cal_jsd else kl_div


def plot_group_Multi(groupdata, x_labels=None, precisions=None, bound=None, m_pt=[15.85, 50, 84.15], \
    c_pt=68.3, true_values=None, colors=None, labels=None, textloc=[0.75, 0.90, 0.00, 0.025], \
    title_size=11, legend_size=30, filename=None):
    """Plot groups of multi-dim data.

    Parameters
    ----------
    groupdata: list
        groups of samples to compare
    x_labels: list of string, optional
        name of every dimension of samples
    precisions: list of int, optional
        precision of title shown
    bound: list, optional
        [[low], [up]] limits of each parameters
    colors: list
        colors of each sample
    m_pt: list, default is [15.85, 50, 84.15] 
        percentiles of every dimension of samples
    c_pt: float, default is 68.3
        contour percentile
    true_values: list, optional
        injected values
    labels: list, optional
        tag of each group
    textloc: list, request only if labels is not None
        control where to put the text, format is [xbegin, ybegin, xshift, yshift]
    filename: str
        save to path 'filename' if given, show it directly it is default value
    
    Returns
    -------
        None
    """
    import numpy as np
    from pycbc.results.scatter_histograms import create_multidim_plot
    inner_key = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', \
                 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    tot_groups = len(groupdata)
    dims = len(groupdata[0])
    select_key = inner_key[0:dims]
    group_samples = [dict(zip(select_key, data)) for data in groupdata]
    if bound==None:
        temp_mins = {key:min([min(samples[key]) for samples in group_samples]) for key in select_key}
        temp_maxs = {key:max([max(samples[key]) for samples in group_samples]) for key in select_key}
        shifts = {key:(temp_maxs[key]-temp_mins[key])/100. for key in select_key}
        mins = {key:temp_mins[key]-shifts[key] for key in select_key}
        maxs = {key:temp_maxs[key]+shifts[key] for key in select_key}
    else:
        mins = dict(zip(select_key, bound[0]))
        maxs = dict(zip(select_key, bound[1]))
    if true_values is not None:
        true_values = dict(zip(select_key, true_values))
    if x_labels is not None:
        x_labels = dict(zip(select_key, x_labels))
    
    fig, axes = None, None
    if colors is None:
        colors = ['g', 'b', 'k', 'cyan', 'purple', 'pink']
    if precisions is None:
        precisions = [2 for i in range(tot_groups)]
    locs = [['left', 'center', 'right'], ['center', 'left', 'right'], ['left', 'right', 'center']]
    locs = locs[tot_groups%3]
    if len(textloc)==4:
        use_patch = False
        x_begin, y_begin, x_shift, y_shift = textloc
    else:
        use_patch = True
        from matplotlib.patches import Patch
        patches = [Patch(color=c) for c in colors]
    for j,samples in enumerate(group_samples):
        fig, axes = create_multidim_plot(select_key, samples, labels=x_labels, 
        mins=mins, maxs=maxs, expected_parameters=true_values, expected_parameters_color='r', 
        plot_marginal=True, plot_scatter=False, marginal_percentiles=m_pt, 
        contour_percentiles=[c_pt], marginal_title=False, marginal_linestyle='-', 
        zvals=None, show_colorbar=False, cbar_label=None, vmin=None, vmax=None, 
        scatter_cmap='plasma', plot_density=False, plot_contours=True, density_cmap='BuPu', 
        contour_color=colors[j], hist_color=colors[j], line_color=colors[j], fill_color=None,
        use_kombine=False, fig=fig, axis_dict=axes)
        for i, (key, precision) in enumerate(zip(select_key, precisions)):
            if i==0:
                axes[(key, key)][0].set_ylabel('PDF')
            probs = np.percentile(samples[key], m_pt)
            temp_axis = axes[(key, key)][0].twiny()
            if precision<=0:
                format_string = r"${:.0f}_{{-{:.0f}}}^{{+{:.0f}}}$"
            else:
                format_string = r"${:."+str(precision)+r"f}_{{-{:."+str(precision)+r"f}}}^{{+{:."+str(precision)+r"f}}}$"
            title_str = format_string.format(np.round(probs[1], precision), np.round(probs[1]-probs[0], precision), np.round(probs[2]-probs[1], precision))
            temp_axis.set_title(title_str, fontsize=title_size, color=colors[j], pad=5+(j/3)*15, loc=locs[j%3])
            temp_axis.set_xticks([])
            axes[(key, key)][0].set_xlim(mins[key], maxs[key])
            axes[(key, key)][0].tick_params(direction='in', labelsize=10)
        if labels is not None:
            if use_patch:
                pass
            else:
                plt.annotate(labels[j], xy=(x_begin-x_shift*j, y_begin-y_shift*j), xycoords='figure fraction', \
                    color=colors[j], fontsize=legend_size)
    if use_patch:
        fig.legend(handles=patches[:dims], loc=textloc, labels=labels[:dims], fontsize=20)
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_corr(samples, names=None, cluster=True, annot=True, figsize=(10, 10), \
              metric_par=(2, 4), cmap="Blues", filename=None, mask_sym=True, **kargs):
    """Plot cross correlation map.

    Parameters
    ----------

    samples: array like
        samples to calculate correlation heat/cluster map
    names: list of string, optional
        name of every dimension of samples
    cluster: bool, optional
        plot a cluster map or heat map
    anoot: bool, optional
        whether to note the correlation number
    figsize: tuple, optional
        figure size 
    metric_par: tuple of int, optional
        the metric is 'sin(t)^a*(sum_i{abs(u_i)-abs(v_i)})^b'
    cmap: cmap 
        cmap of matplotlib
    filename: str
        save to path 'filename' if given, show it directly it is default value
    mask_sym: bool, optional
        whether to mask the symmetry part
    kargs: dict
        transfered to heat map or cluster map
    
    Returns
    -------
        None
    """
    import seaborn as sbn
    
    def my_metric(u, v):
         nu = np.linalg.norm(u)
         nv = np.linalg.norm(v)
         scale_fac = np.linalg.norm(np.abs(u)-np.abs(v))
         sin_theta = np.sqrt(1-(np.dot(u, v)/(nu*nv))**2)
         return np.power(sin_theta, metric_par[0])*scale_fac**metric_par[1]

    if not hasattr(names, '__iter__'):
        names = [str(i+1) for i in range(len(samples))]
    corr = np.corrcoef(samples)
    print("Parameter names:\n\t {}".format(names))
    print("Cross correlation coefficients:\n")
    print("{}".format(corr))
    kargs.update({"xticklabels":names, "yticklabels":names})
    if mask_sym:
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask, 1)] = True
    else:
        mask = None
    if cluster:
        sbn.clustermap(corr, cmap=cmap, annot=annot, metric=my_metric, mask=mask, figsize=figsize, **kargs)
    else:
        with sbn.axes_style("white"):
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()
            sbn.heatmap(corr, cmap=cmap, annot=annot, ax=ax, mask=mask, **kargs)
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)


def compare_pdf_plots(funcs, xbound, conf_interval=90., labels=None, xlabel='x', \
                      ylabel='PDF', colors=None, filename=None, lw=2, show_vlines=True):
    """Compare multi pdfs.

    Parameters
    ----------
    funcs: list of univariate normed functions
    xrange: (xmin, xmax) tuple of double
    percentiles: list of float
        Percentiles to be shown.
    labels: labels to show
    filename: str
        save to path 'filename' if given, show it directly it is default value
    """

    if not hasattr(labels, '__iter__'):
        labels = np.arange(1, len(funcs)+1)
    x = np.linspace(xbound[0], xbound[1], 20000)
    y_max = 0
    for i, (lb, func) in enumerate(zip(labels, funcs)):
        y = func(x)
        ym = max(y)
        y_max = max(ym, y_max)
        if hasattr(colors, '__iter__'):
            color = colors[i]
        else:
            color = (np.random.rand(), np.random.rand(), np.random.rand())
            print("You have chose color {} for line {}".format(color, i+1))
        perc = cal_pdf_property(x, y, conf_interval, True, False, 6, False)
        plt.plot(x, y, c=color, label=str(lb), lw=lw)
        if show_vlines:
            plt.vlines(perc, 0, ym, colors=color, lw=0.8, linestyle='--')
    ybound = (0, y_max*1.1)
    plt.xlim(xbound)
    plt.ylim(ybound)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)

def compare_sample_plots(samples, percentiles=[5., 50., 95.], labels=None, xlabel='x', \
                         xmin=None, xmax=None, ylabel='PDF', colors=None, filename=None, \
                         lw=2, lss=None, show_vlines=True, bw_method="scott"):
    """Compare multi samples.

    Parameters
    ----------
    percentiles: list of float
        Percentiles to be shown.
    labels: labels to show
    filename: str
        save to path 'filename' if given, show it directly it is default value
    """

    from scipy.stats import gaussian_kde

    samples = np.array(samples)
    if labels is None:
        labels = np.arange(1, len(samples)+1)
    if colors is None:
        colors = my_beauty_colors[:]
    if lss is None:
        lss = ['-']*len(samples)
    mmin = min([min(s) for s in samples])
    mmax = max([max(s) for s in samples])
    if (xmin is not None) and (xmax is not None):
        x = np.linspace(xmin, xmax, 2000)
    else:
        x = np.linspace(mmin, mmax, 2000)
    y_max = 0
    for i, (lb, ls, sample) in enumerate(zip(labels, lss, samples)):
        kde = gaussian_kde(sample, bw_method=bw_method)
        y = kde(x)
        ym = max(y)
        y_max = max(ym, y_max)
        perc = [np.percentile(sample, p) for p in percentiles]
        plt.plot(x, y, c=colors[i], label=str(lb), ls=ls, lw=lw)
        if show_vlines:
            plt.vlines(perc, 0, ym, colors=colors[i], lw=0.8, linestyle='--')
    fac = 1.1
    if xmin is None:
        xmin = mmin*fac if np.sign(mmin)==-1 else mmin/fac
    if xmax is None:
        xmax = mmax/fac if np.sign(mmax)==-1 else mmax*fac
    xbound = (xmin, xmax)
    ybound = (0, y_max*1.1)
    if np.shape(samples)[0]==3:
        perc1 = [np.percentile(samples[0], p) for p in percentiles]
        perc2 = [np.percentile(samples[1], p) for p in percentiles]
        perc3 = [np.percentile(samples[2], p) for p in percentiles]
        title1 = r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(perc1[1], perc1[1]-perc1[0], \
             perc1[2]-perc1[1])
        title2 = r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(perc2[1], perc2[1]-perc2[0], \
             perc2[2]-perc2[1])
        title3 = r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(perc3[1], perc3[1]-perc3[0], \
             perc3[2]-perc3[1])
        plt.title(title1, color=colors[0], loc='left')
        plt.title(title2, color=colors[1], loc='center')
        plt.title(title3, color=colors[2], loc='right')
    plt.xlim(xbound)
    plt.ylim(ybound)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)


def compare_two_sample_stats(sample1, sample2):
    """Compare stats of two samples.
    """

    from scipy.stats import ks_2samp, mannwhitneyu
    from statsmodels.sandbox.stats.runs import runstest_2samp

    js_divergence = cal_kldiv(sample1, sample2, cal_jsd=True)
    D, p_value = ks_2samp(sample1, sample2)
    rank_z, rank_p = mannwhitneyu(sample1, sample2)
    run_z, run_p = runstest_2samp(sample1, sample2)
    print("Jensen-Shannon divergence: {}".format(js_divergence))
    print("\n----------------------------------------------------------------------------------\n")
    print("    Test method                                            statistic      p-value")
    print("======================================================     ==========   ==========")
    print("Two sided Kolmogorov-Smirnov test                     :    {:<+.3E}\t{:<+.3E}".format(D, p_value))
    print("(sensitive to both the diff of mean and variance)")
    print("Wilcoxon-Mann-Whitney rank test                       :    {:<+.3E}\t{:<+.3E}".format(rank_z,rank_p))
    print("(more sensitive to the diff of mean but not variance)")
    print("Wald-Wolfowitz run test                               :    {:<+.3E}\t{:<+.3E}".format(run_z,run_p))
    print("(more sensitive to the diff of variance but not mean)")

def plot_crude_diff(lmh1, lmh2, lb1='1', lb2='2', xlabel='x', filename=None):
    """
    filename: str
        save to path 'filename' if given, show it directly it is default value
    """

    import mcerp
    mmin, mmax = min(lmh1[0], lmh2[0]), max(lmh1[2], lmh2[2])
    dist1 = mcerp.PERT(lmh1[0], lmh1[1], lmh1[2])
    dist2 = mcerp.PERT(lmh2[0], lmh2[1], lmh2[2])
    dist1.plot(label=lb1)
    dist2.plot(label=lb2)
    plt.xlabel(xlabel)
    plt.xlim(mmin, mmax)
    plt.ylim(0, None)
    plt.legend()
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)


def compare_two_sample_plots(sample1, sample2, percentiles=[5., 50., 95.], \
                             xlabel='x', xscale='linear', xmin=None, xmax=None, \
                             bins='auto', histtype='bar', c1=None, c2=None, name1=None, \
                             name2=None, filename=None, show_hist=False):
    """Compare two samples.

    Parameters
    ----------
    percentiles: list of float
        Percentiles to be shown.
    bins: bins
        see numpy.histogram
    histtype: histtype
        see pyplot.hist
    filename: str
        save to path 'filename' if given, show it directly it is default value
    """

    from scipy.stats import gaussian_kde

    mmin = min(np.hstack((sample1, sample2)))
    mmax = max(np.hstack((sample1, sample2)))
    x = np.linspace(mmin, mmax, 2000)
    kde1 = gaussian_kde(sample1, bw_method="scott")
    kde2 = gaussian_kde(sample2, bw_method="scott")
    y1 = kde1(x)
    y2 = kde2(x)
    perc1 = [np.percentile(sample1, p) for p in percentiles]
    perc2 = [np.percentile(sample2, p) for p in percentiles]
    y_max = max(np.hstack((y1, y2)))
    fac = 1.2
    if xmin==None:
        xmin = mmin*fac if np.sign(mmin)==-1 else mmin/fac
    if xmax==None:
        xmax = mmax/fac if np.sign(mmax)==-1 else mmax*fac
    xbound = (xmin, xmax)
    ybound = (0, y_max*1.3)
    plt.xlim(xbound)
    #plt.ylim(ybound)
    if c1==None:
        c1 = (np.random.rand(), np.random.rand(), np.random.rand())
    if c2==None:
        c2 = (np.random.rand(), np.random.rand(), np.random.rand())
    label1 = "kde of sample1" if name1==None else "kde of "+name1
    label2 = "kde of sample2" if name2==None else "kde of "+name2
    plt.plot(x, y1, c=c1, label=label1, lw=2)
    plt.plot(x, y2, c=c2, label=label2, lw=2)
    plt.vlines(perc1, 0, y_max, colors=c1, lw=1, linestyle='--')
    plt.vlines(perc2, 0, y_max, colors=c2, lw=1, linestyle='--')
    if show_hist:
        label1 = "hist of sample1" if name1==None else "hist of "+name1
        label2 = "hist of sample2" if name2==None else "hist of "+name2
        if xscale=='log':
            _, bins1 = np.histogram(np.log10(sample1[sample1!=0.]), bins=bins)
            _, bins2 = np.histogram(np.log10(sample2[sample2!=0.]), bins=bins)
            plt.hist(sample1, bins=10**bins1, density=True, label=label1, 
                alpha=0.7, color=c1, histtype=histtype)
            plt.hist(sample2, bins=10**bins2, density=True, label=label2,
                alpha=0.7, color=c2, histtype=histtype)
            xlabel = 'log10('+xlabel+')'
        else:
            plt.hist(sample1, bins=bins, density=True, label=label1,
                alpha=0.5, color=c1, histtype=histtype)
            plt.hist(sample2, bins=bins, density=True, label=label2,
                alpha=0.5, color=c2, histtype=histtype)
    title1 = r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(perc1[1], perc1[1]-perc1[0], \
             perc1[2]-perc1[1])
    title2 = r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(perc2[1], perc2[1]-perc2[0], \
             perc2[2]-perc2[1])
    plt.title(title1, color=c1, loc='left')
    plt.title(title2, color=c2, loc='right')
    plt.ylabel("PDF")
    plt.xlabel(xlabel)
    plt.xscale(xscale)
    plt.legend()
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)


def compare_two_samples(sample1, sample2, filename=None):
    
    """
    filename: str
        save to path 'filename' if given, show it directly it is default value
    """
    compare_two_sample_stats(sample1, sample2)
    prop1 = np.array(cal_sample_property(sample1, method="median"))
    prop2 = np.array(cal_sample_property(sample2, method="median"))
    diff_abs = prop1-prop2
    diff_percent = diff_abs/prop2*100
    print("-----------------------------------------------------------------")
    print("sample1    lower: {:<+.3E} median: {:<+.3E} upper: {:<+.3E}".\
          format(prop1[0], prop1[1], prop1[2]))
    print("sample2    lower: {:<+.3E} median: {:<+.3E} upper: {:<+.3E}".\
          format(prop2[0], prop2[1], prop2[2]))
    print("diff(1-2)  lower: {:<+.3E} median: {:<+.3E} upper: {:<+.3E}".\
          format(diff_abs[0], diff_abs[1], diff_abs[2]))
    print("((1-2)/2)% lower: {:<+.3E} median: {:<+.3E} upper: {:<+.3E}".\
          format(diff_percent[0], diff_percent[1], diff_percent[2]))
    compare_two_sample_plots(sample1, sample2, filename=filename)


def compare_two_sample_pairs(pair1, pair2, xlabel=None, ylabel=None, colors=['c', 'm'], \
                             lower_1=None, upper_1=None, lower_2=None, upper_2=None, \
                             m_pt=[5, 50, 95], c_pt=[68.3, 95.5], plot_config=None, \
                             filename=None):
    """Compare two variable density plots using PyCBC.

    Parameters
    ----------
    pair1: array of shape (2, N)
        first pair of array to show
    pair2: array of shape (2, N)
        second pair of array to show
    m_pt: list of float 
        marginal_percentiles
    c_pt: list of float  
        contour_percentiles
    filename: str
        save to path 'filename' if given, show it directly it is default value
    """

    from pycbc.results.scatter_histograms import create_multidim_plot

    if xlabel==None: xlabel='x'
    if ylabel==None: ylabel='y'
    if plot_config==None:
        plot_config = {'plot_contours': True, 'show_colorbar':False, \
                       'plot_scatter': False, 'mins': {xlabel:lower_1, ylabel: \
                       lower_2}, 'maxs': {xlabel:upper_1, ylabel:upper_2}, \
                       'density_cmap': 'BuPu', 'contour_percentiles': c_pt, \
                       'marginal_percentiles': m_pt, 'fill_color':None}
    samples = [{xlabel:pair1[0], ylabel:pair1[1]}, {xlabel:pair2[0], ylabel:pair2[1]}]
    for (i, sample) in enumerate(samples):
        if i == 0:
            fig = None
            axis_dict = None
        color = colors[i]#colors.next()
        fig, axis_dict = create_multidim_plot(sample.keys(), sample, fig=fig, \
                         axis_dict = axis_dict, contour_color=color, \
                         hist_color = color, line_color=color, **plot_config)
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)


def dynamic_single(sample, fixed_sample=None, frames=20, wait_time=0.5, \
                   xbound=None, scale_fac=1.2, bins='auto', percentiles=[5., 50., 95.]):
    """Show how a sample varies with time.

    Parameters
    ----------
    sample: array of shape (1, N)
        Sample that to be shown dynamically.
    fixed_sample: array, optional
        Sample to compare, do not vary with time.
    frames: int, optional
        Frame number of gif.
    wait_time: float, optional
        Time to wait per frame(in unit of second).
    xbound: tuple of float
        X axis range, in the shape of (lower, upper).
    scale_fac: float, optional
        Factor to scale x,y range.
    percentiles: list of float, optional
        Percentiles to be shown.
    """

    from scipy.stats import gaussian_kde

    length = len(sample)
    first = length/5
    indexes = [int(i) for i in np.linspace(first, length, frames)]+[length]
    if hasattr(fixed_sample, '__iter__'):
        total_sample = np.hstack((sample, fixed_sample))
        mmin = min(total_sample)
        mmax = max(total_sample)
        x = np.linspace(mmin, mmax, 2000)
        fix_kde = gaussian_kde(fixed_sample, bw_method="scott")
        y_fix = fix_kde(x)
        perc_fix = [np.percentile(fixed_sample, p) for p in percentiles]
    else:
        mmin = min(sample)
        mmax = max(sample)
        x = np.linspace(mmin, mmax, 2000)
        total_sample = sample
    total_kde = gaussian_kde(total_sample, bw_method="scott")
    y_max = max([total_kde(i) for i in x])
    if not hasattr(xbound, '__iter__'):
        xmin = mmin*scale_fac if np.sign(mmin)==-1 else mmin/scale_fac
        xmax = mmax/scale_fac if np.sign(mmax)==-1 else mmax*scale_fac
        xbound = (xmin, xmax)
    ybound = (0, y_max*scale_fac)
    plt.ion()
    for idx in indexes:
        take = sample[:idx]
        kde = gaussian_kde(take, bw_method="scott")
        y = kde(x)
        perc = [np.percentile(take, p) for p in percentiles]
        plt.cla()
        plt.xlim(xbound)
        plt.ylim(ybound)
        plt.plot(x, y, c='g', label="dynamic kde", lw=2)
        plt.vlines(perc, 0, y_max, colors='g', lw=1, linestyle='--')
        label = "dynamic hist"
        plt.hist(take, bins=bins, density=True, label=label, alpha=0.8)
        if hasattr(fixed_sample, '__iter__'):
            plt.vlines(perc_fix, 0, y_max, colors='r', lw=1, linestyle='--')
            plt.plot(x, y_fix, c='r', label="reference kde", lw=2)
            plt.hist(fixed_sample, bins=bins, density=True, alpha=0.8,\
                     label= "reference hist")
        plt.ylabel("PDF")
        plt.legend()
        plt.pause(wait_time)
    plt.pause(4)
    plt.close()


def dynamic_walkers(sample, binned_steps=2, wait_time=0.5, xbound=None, \
                    scale_fac=1.2, bins='auto', percentiles=[5., 50., 95.]):
    """Show how a MCMC sample with multiple walkers varies with time.

    Parameters
    ----------
    sample: array of shape (walker, steps)
        Sample that to be shown dynamically.
    binned_steps: array, optional
        Every binned_steps steps will be binned together to show.
    frames: int, optional
        Frame number of gif.
    wait_time: float, optional
        Time to wait per frame(in unit of second).
    xbound: tuple of float
        X axis range, in the shape of (lower, upper).
    scale_fac: float, optional
        Factor to scale x,y range.
    percentiles: list of float, optional
        Percentiles to be shown.
    """

    from scipy.stats import gaussian_kde

    sample = np.array(sample)
    walkers = sample.shape[0]
    steps = sample.shape[1]
    indexes = [i for i in range(binned_steps, steps, binned_steps)]+[steps]
    mmin = min(sample.flatten())
    mmax = max(sample.flatten())
    x = np.linspace(mmin, mmax, 2000)
    t_kde = gaussian_kde(sample[:,:binned_steps].flatten(), bw_method="scott")
    y_max = max([t_kde(i) for i in x])
    if not hasattr(xbound, '__iter__'):
        xmin = mmin*scale_fac if np.sign(mmin)==-1 else mmin/scale_fac
        xmax = mmax/scale_fac if np.sign(mmax)==-1 else mmax*scale_fac
        xbound = (xmin, xmax)
    ybound = (0, y_max*scale_fac**2)
    plt.ion()
    for idx in indexes:
        if binned_steps==1:
            take = sample[:, idx-1].flatten()
        else:
            lower_idx = idx-binned_steps+1
            take = sample[:, lower_idx:idx].flatten()
        kde = gaussian_kde(take, bw_method="scott")
        y = kde(x)
        perc = [np.percentile(take, p) for p in percentiles]
        plt.cla()
        plt.xlim(xbound)
        plt.ylim(ybound)
        plt.plot(x, y, c='g', label="kde", lw=2)
        plt.vlines(perc, 0, y_max, colors='g', lw=1, linestyle='--')
        if binned_steps==1:
            plt.title("steps:{} x walkers:{}".format(idx, walkers))
        else:
            plt.title("steps:{}-{} x walkers:{}".format(lower_idx, idx, walkers))
        plt.hist(take, bins=bins, density=True, label="hist", alpha=0.8)
        plt.ylabel("PDF")
        plt.legend()
        plt.pause(wait_time)
    plt.pause(4)
    plt.close()


#============================================================================#
#                                                                            #
#                         Parameter Transformations                          #
#                                                                            #
#============================================================================#


def get_lt_prior_from_m12l12(size, m1_l, m1_u, l1_l, l1_u, m2_l=None, \
    m2_u=None, l2_l=None, l2_u=None, mc_l=0, mc_u=np.inf, q_l=1., q_u=np.inf):
    """Get lambda tilde prior sample from uniform parameters(m1, m2, l1, l2), 
    and constraints, parameters set to none will be set equal to similar '1' 
    parameter by default, egg., if m2_l is not given, it will be set equal to 
    m1_l by default.

    Parameters
    ----------
    size: int
        the sample size wanted.
    'x'_l: float
        lower limit of parameter 'x'.
    'x'_u: float
        upper limit of parameter 'x'.

    Returns
    -------
    prior sample of lambda tilde: list
    """

    from pycbc.distributions import Uniform, JointDistribution
    from pycbc.conversions import mchirp_from_mass1_mass2 as mc_from_m12
    from pycbc.conversions import q_from_mass1_mass2 as q_from_m12

    def constraint(param):
        mc = mc_from_m12(param["mass1"], param["mass2"])
        q = q_from_m12(param["mass1"], param["mass2"])
        return (mc>mc_l)*(mc<mc_u)*(q>q_l)*(q<q_u)

    if m2_l==None: m2_l=m1_l
    if m2_u==None: m2_u=m1_u
    if l2_l==None: l2_l=l1_l
    if l2_u==None: l2_u=l1_u
    prior = Uniform(mass1=(m1_l, m1_u), mass2=(m2_l, m2_u), \
                    lambda1=(l1_l, l1_u), lambda2=(l2_l, l2_u))
    varg = ['mass1', 'mass2', 'lambda1', 'lambda2']
    joint = JointDistribution(varg, prior, constraints=[constraint]).rvs(size)
    lt_prior = [lambda_tilde(mass1=joint[i][0],mass2=joint[i][1], \
                lambda1=joint[i][2],lambda2=joint[i][3]) for i in range(size)]
    return lt_prior


def get_lt_prior_from_mcql12(size, mc_l, mc_u, q_l, q_u, l1_l, l1_u, \
             m1_l=0, m1_u=np.inf, m2_l=None, m2_u=None, l2_l=None, l2_u=None):
    """Get lambda tilde prior sample from uniform parameters(mc, q, l1, l2), 
    and constraints, parameters set to none will be set equal to similar '1' 
    parameter by default, egg., if m2_l is not given, it will be set equal to 
    m1_l by default.

    Parameters
    ----------
    size: int
        the sample size wanted.
    'x'_l: float
        lower limit of parameter 'x'.
    'x'_u: float
        upper limit of parameter 'x'.

    Returns
    -------
    prior sample of lambda tilde: list
    """

    from pycbc.distributions import Uniform, JointDistribution

    def constraint(param):
        #print(param["mc"], param['q'])
        m1, m2 = m12_from_mcq(param["mc"], param["q"])
        return (m1<m1_u)*(m1>m1_l)*(m2<m2_u)*(m2>m2_l)

    if m2_l==None: m2_l=m1_l
    if m2_u==None: m2_u=m1_u
    if l2_l==None: l2_l=l1_l
    if l2_u==None: l2_u=l1_u
    prior = Uniform(mc=(mc_l, mc_u), q=(q_l, q_u), \
                    lambda1=(l1_l, l1_u), lambda2=(l2_l, l2_u))
    varg = ['mc', 'q', 'lambda1', 'lambda2']
    joint = JointDistribution(varg, prior, constraints=[constraint]).rvs(size)
    lt_prior = [lambda_tilde(*m12_from_mcq(joint[i][0], joint[i][1]), \
                lambda1=joint[i][2], lambda2=joint[i][3]) for i in range(size)]
    return lt_prior


def plot_mass_range(m1_range, mc_range, q_range, m2_range=None, filename=None):
    """Show allowed range of mass given constraint of mass1, mass2, chirp
    mass and mass ratio, a range should be a tuple like (lower, upper) 

    Parameters
    ----------
    m1_range: tuple of float
        range of mass1
    mc_range: tuple of float
        range of chirp mass
    q_range: tuple of float
        range of mass ratio
    m2_range: tuple of float, optional
        range of mass2, set to m1_range by default
    filename: str
        save to path 'filename' if given, show it directly it is default value
    """
    from pycbc.conversions import mass1_from_mchirp_q as m1_from_mcq
    from pycbc.conversions import mass2_from_mchirp_q as m2_from_mcq

    if not hasattr(m2_range, '__iter__'): m2_range = m1_range
    m1 = []
    m2 = []
    for mc in np.linspace(mc_range[0], mc_range[1], 300):
        for q in np.linspace(q_range[0], q_range[1], 300):
            m1.append(m1_from_mcq(mc, q))
            m2.append(m2_from_mcq(mc, q))
    m1 = np.array(m1)
    m2 = np.array(m2)
    take_indexes = (m1>m1_range[0])*(m1<m1_range[1])* \
                   (m2>m2_range[0])*(m2<m2_range[1])
    not_take_indexes = [not item for item in take_indexes]
    m1_take = m1[take_indexes]
    m2_take = m2[take_indexes]
    m1_not_take = m1[not_take_indexes]
    m2_not_take = m2[not_take_indexes]
    mt = m1_take+m2_take
    cs = plt.scatter(m1_take, m2_take, c=mt, label="allowed mass range", s=2)
    plt.scatter(m1_not_take, m2_not_take, c='k', \
                             label="mc-q allowed but m1-m2 not allowed", s=2)
    plt.vlines(m1_range, m2_range[0], m2_range[1], color='r', lw=2, \
                                                   label="mass prior range")
    plt.hlines(m2_range, m1_range[0], m1_range[1], color='r', lw=2)
    plt.xlabel(r"$m_1/M_{\odot}$")
    plt.ylabel(r"$m_2/M_{\odot}$")
    cb = plt.colorbar(cs, ticks=np.linspace(min(mt), max(mt), 20))
    cb.set_label(r"$M_{tot}/M_{\odot}$")
    plt.legend()
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)


def mclt_from_m12l12(mass1, mass2, lambda1, lambda2):
    from pycbc.conversions import mchirp_from_mass1_mass2 as mc_from_m12

    mc = mc_from_m12(mass1, mass2)
    lt = lambda_tilde(mass1, mass2, lambda1, lambda2)
    return mc, lt


def mcq_from_m12(mass1, mass2):
    from pycbc.conversions import q_from_mass1_mass2 as q_from_m12
    from pycbc.conversions import mchirp_from_mass1_mass2 as mc_from_m12

    mc = mc_from_m12(mass1, mass2)
    q = q_from_m12(mass1, mass2)
    return mc, q


def m12_from_mcq(mchirp, q):
	mass1 = (q**(2./5.))*((1.0 + q)**(1./5.))*mchirp
	mass2 = (q**(-3./5.))*((1.0 + q)**(1./5.))*mchirp
	return mass1, mass2


def lambda_tilde(mass1, mass2, lambda1, lambda2):
    return (16./13.)*((mass1+12.*mass2)*mass1**4*lambda1+(mass2+12.*mass1)*mass2**4*lambda2)/(mass1+mass2)**5


def q12_from_qtm(qt, qm, root_choose='+'):
    q1 = (qm+np.sqrt(qm**2+4*qt))/2., (qm-np.sqrt(qm**2+4*qt))/2.
    q2 = q1[0]-qm, q1[1]-qm
    return (q1[0], q2[0]) if root_choose=='+' else (q1[1], q2[1])


#============================================================================#
#                                                                            #
#                                 Data Io                                    #
#                                                                            #
#============================================================================#




def fp(fname, io_state='r', type='hdf'):
    if type=='hdf':
        import h5py
        return h5py.File(fname, io_state)
    elif type=='json':
        import json
        return json.load(open(fname, io_state))
    else:
        print("Data type not recognized!")


def get_txt_pars(input_file, delimiter=None):
    """Get parameter names in a txt file using the first line.
    
    Parameters
    ----------
    input_file: string
        Name of the input txt file.
    delimiter: string, optional
        Characters to split title.

    Returns
    -------
    list of strings
    """

    with open(input_file, "r") as fp:
        para_names = fp.readline().lstrip('#').strip().split(delimiter)
    return para_names


def get_hdf_pars(fpath, param_path="/data/posterior", data_type="dataframe"):
    """Read hdf data.

    Parameters
    ----------
    fpath: string
        Path to file.
    param_path: string
        Group name of parameter position, should not include trailing '/'.
    data_type: string, optional ["dataset", "dict", "dataframe"]
        The way that data is structured.

    Returns
    -------
    list of strings
    """

    import os
    import h5py

    if os.path.isfile(fpath):
        fp = h5py.File(fpath, 'r')
        if data_type=="dataset":
            params_list = fp[param_path].keys()
        elif data_type=="dict":
            params_list = fp[param_path].dtype.names
        elif data_type=="dataframe":
            params_list = fp[param_path]["block1_items"].value
        else:
            print("Unkown data_type!")
        fp.close()
    else:
        print("No such fucking file \n" * 10)
        raise IOError
    return params_list


def load_txt(input_file, params_list=None, delimiter=' '):
    """Load data from a txt file.
    
    Parameters
    ----------
    input_file: string
        Name of the input txt file.
    params_list: string, optional
        List of the parameters needed to load from the txt file.
    delimiter: string, optional
        Characters to split title.

    Returns
    -------
    numpy.ndarray
    """

    with open(input_file, "r") as fp:
        pars = np.array(fp.readline().strip().split(delimiter))
        fparams = [p.strip() for p in pars if p.strip()]
    if not hasattr(params_list, '__iter__'):
        posit = {param:i for i, param in enumerate(fparams)}
        params_list = fparams
        print(fparams)
    else:
        posit = {param:np.where(fparams==param)[0][0] for param in params_list}
    data = np.loadtxt(input_file, skiprows=1).T
    wanted = []
    for param in params_list:
        wanted.append(data[posit[param]])
    return np.array(wanted)


def load_hdf(fpath, data_path="/samples", params_list=None, \
             data_type="pycbc", pt_sampler=False, return_dict=False):
    """Read hdf data.

    Parameters
    ----------
    fpath: string
        Path to file.
    data_path: string
        Group name of wanted dataset, should not include trailing '/'.
    params_list: list of string, optional
        Parameters wanted.
    data_type: string, optional 
        can be either ["dataset", "dict", "dataframe"], or ["pycbc", "ligo", "bilby"].
        The way that data is structured.
    result_type: string, optional 
        How did you get the result file.
    pt_sampler: bool, optional
        Whether the results are sampled by pt sampler, set to True to read samples at
        zero temperature.

    Returns
    -------
    numpy.ndarray
    """

    import os
    import h5py

    retn = []
    if os.path.isfile(fpath):
        fp = h5py.File(fpath, 'r')
        if (data_type=="dataset" or data_type=="pycbc"):
            if not hasattr(params_list, '__iter__'):
                params_list = fp[data_path].keys()
                print(params_list)
            for param in params_list:
                if pt_sampler:
                    toappend = np.array(fp[data_path+'/'+param][0])
                else:
                    toappend = np.array(fp[data_path+'/'+param])
                retn.append(toappend)
        elif (data_type=="dict" or data_type=="ligo"):
            if not hasattr(params_list, '__iter__'):
                params_list = fp[data_path].dtype.names
                print(params_list)
            for param in params_list:
                retn.append(np.array(fp[data_path][param]))
        elif (data_type=="dataframe" or data_type=="bilby"):
            names = np.array(fp[data_path]["block1_items"])
            all_data = np.array(fp[data_path]["block1_values"]).T
            retn = np.vstack((all_data[names==name] for name in params_list))
        else:
            print("Unkown data_type!")
        fp.close()
        retn = np.array(retn)
    else:
        print("No such fucking file \n" * 10)
        raise IOError
    if return_dict:
        retn = {k:retn[i] for i, k in enumerate(params_list)}
    return retn


def load_thin_hdf(fpath, data_path, params, pt_sampler=True, flatten=True):
    """load data from a hdf file.

    Parameters
    ----------
    fpath: string
        Name of the input hdf file
    data_path: string
        path of wanted data
    params: string
        List of the parameters needed to load from the hdf file
    pt_sampler: bool, optional
        Whether the results are generated by pt_sampler.
    flatten: bool, optional
        Whether to return a flattened array.

    Returns
    -------
    numpy.ndarray
    """
    import h5py
    import numpy as np

    fp = h5py.File(fpath, "r")
    if pt_sampler:
        unthin = {param:fp[data_path][param][0] for param in params}
    else:
        unthin = {param:fp[data_path][param] for param in params}
    start = fp.attrs["thin_start"]
    interval = fp.attrs["thin_interval"]
    end = None
    if "thin_end" in fp.attrs.keys():
        end = fp.attrs["thin_end"]
    fp.close()
    wanted = []
    for param in params:
        to_append = np.array(unthin[param][:, start:end:interval])
        if flatten:
            to_append = to_append.flatten()
        wanted.append(to_append)
    wanted = np.array(wanted)
    return wanted


def write_txt(filename, data_list, title_list, fill_width=10):
    row_num = len(data_list[0])
    column_num = len(title_list)
    title_string = ""
    data_string = "\n"
    for i in range(column_num):
        title_string += "{:<."+str(fill_width)+"}\t"
        data_string += "{:<."+str(fill_width)+"}\t"
    with open(filename,'w') as f:
        f.write(title_string.format(*title_list))
        for i in range(row_num):
            f.write(data_string.format(*[list[i] for list in data_list]))


#============================================================================#
#                                                                            #
#                                  Other Utils                               #
#                                                                            #
#============================================================================#


useful_centences = ["from pycbc.waveform.compress import rough_time_estimate", \
                    "from scipy.stats import pearsonr"]


from scipy.stats import rv_continuous

class kde_data(rv_continuous):
    def __init__(self, data, lower=None, upper=None, **kargs):
        """ Re-sample data from kde of a group of data 
        """

        from scipy.integrate import quad
        from scipy.stats import gaussian_kde

        self.lower = min(data) if lower==None else lower
        self.upper = max(data) if upper==None else upper
        self.kde = gaussian_kde(data)
        self.scale = quad(self.kde, self.lower, self.upper)[0]
        rv_continuous.__init__(self, a=self.lower, b=self.upper, **kargs)
        
    def _pdf(self, x):
        return self.kde(x)/self.scale


def asymmetry_dist(d_lower, mean, d_upper, conf_interval=68.26):
    """
    68% lower, mean, upper of a distribution
    """
    from scipy.stats import norm
    from scipy.optimize import ridder
    from scipy.integrate import quad

    def asym_norm(x, c, d, mean=0.):
        scale = 2./(d*(c+1./c))
        x = x-mean
        if len(np.shape(x))==0:
            ret = scale*norm.pdf(c*x/d) if x<=0 else scale*norm.pdf(x/(d*c))
        else:
            ret = np.ones_like(x)
            for i, xx in enumerate(x):
                if xx<=0:
                    ret[i] = scale*norm.pdf(c*xx/d)
                else:
                    ret[i] = scale*norm.pdf(xx/(d*c))
        return ret

    def int_asymnorm(c, d, lb, ub):
        return quad(lambda x: asym_norm(x, c, d, mean=0.), lb, ub)[0]

    par_c = np.sqrt(d_upper/d_lower) 
    par_d = ridder(lambda d: int_asymnorm(par_c, d, -d_lower, d_upper)-conf_interval/100., \
        min(d_lower, d_upper)/10., max(d_lower, d_upper)*10.)
    ret_func = lambda x: asym_norm(x, par_c, par_d, mean)
    return ret_func


def downsample(array, size):
    """Down sample an 1darray or 2darray to a smaller size.
    """

    from pandas.core.frame import DataFrame 
    
    a_dic = {}
    is_one_dim = (len(np.array(array).shape)==1)
    if is_one_dim:
        a_dic.update({'1':array})
    else:
        for i, a in enumerate(array):
            a_dic.update({str(i):a})
    df = DataFrame(a_dic)
    ret = df.sample(size).values.T
    return ret[0] if is_one_dim else ret


def unique_2darray(array):
    """Unique a 2D-array.
    """
    import sys

    array = np.array(array)
    nrepeated = [True]
    for i, ar in enumerate(array[1:]):
        if np.all(abs(ar-array[i])<sys.float_info.epsilon):
            nrepeated.append(False)
        else:
            nrepeated.append(True)
    nrepeated = np.array(nrepeated)
    return array[nrepeated]


def time_before_merger(f_low, chirp_m):
    return 93.9*(f_low/30.)**(-8./3)*(chirp_m/0.87)**(-5./3)
