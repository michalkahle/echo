import pandas as pd
import numpy as np
from math import sqrt
from functools import partial
import plotnine as p9
import re
import os
import xml.etree.ElementTree as ET

_rows = np.array([chr(x) for x in range(65, 91)] + ['A' + chr(x) for x in range(65, 71)])

def welln2well(wells, form):
    form = int(form)
    if form not in [96, 384, 1536]:
        raise ValueError('Only formats 96, 384 and 1536 supported.')
    n_cols = int(sqrt(form/2*3))

    wells = wells if type(wells) == np.ndarray else np.array(wells, dtype=np.int)
    if np.any(wells >= form) or np.any(wells < 0):
        raise ValueError('welln out of range')
    rr = _rows[wells // n_cols]
    cc = (wells % n_cols + 1).astype(str)
    return np.core.defchararray.add(rr, cc)

welln2well_96 = partial(welln2well, form=96)
welln2well_384 = partial(welln2well, form=384)
welln2well_1536 = partial(welln2well, form=1536)

def well2welln(wells, form):
    form = int(form)
    if form not in [96, 384, 1536]:
        raise ValueError('Only formats 96, 384 and 1536 supported.')
    n_cols = int(sqrt(form/2*3))
    wells = wells if type(wells) == np.ndarray else np.array(wells, dtype=np.str)
    _well_regex = re.compile('^([A-Z]{1,2})(\d{1,2})')
    def _w2wn(well, n_cols):
        match = _well_regex.match(well)
        if not match:
            raise ValueError('Well not recognized: "%s"' % well)
        rr, cc = match.group(1), match.group(2)
        rrn = ord(rr) - 65 if len(rr) == 1 else ord(rr[1]) - 39
        ccn = int(cc) - 1
        return rrn * n_cols + ccn
    _vw2wn = np.vectorize(_w2wn, excluded=('n_cols'))
    wns = _vw2wn(wells, n_cols)
    if np.any(wns >= form) or np.any(wns < 0):
        raise ValueError('welln out of range')
    return wns

well2welln_96 = partial(well2welln, form=96)
well2welln_384 = partial(well2welln, form=384)
well2welln_1536 = partial(well2welln, form=1536)

def plot_picklist(picklist, form, fill='v', **kwargs):
    for barcode, pl in picklist.groupby('s_plate'):
        gg = plot_plate(pl, form, fill=fill, alpha=.5, **kwargs)
        gg += p9.ggtitle(barcode)
        gg.draw()

def read_picklist(fn):
    return pd.read_csv(fn, names=['s_plate', 's_well', 't_plate', 'v', 't_well'])

def write_picklist(df, fn):
    if 's_well' not in df.columns:
        # df['s_well'] = we don't know the format!
        pass
    df[['s_plate', 's_well', 't_plate', 'v', 't_well']].to_csv(fn, header=None, line_terminator='\r\n', index=False)

def read_survey(fn):
    tree = ET.parse(fn)
    root = tree.getroot()
    form = int(re.match('^\d+', root.attrib['name']).group())
    ll = [{'well': child.attrib['n'], 'v': float(child.attrib['vl'])} for child in root]
    df = pd.DataFrame(ll)
    df['welln'] = well2welln(df['well'], form)
    df['barcode'] = root.attrib['barcode']
    df['format'] = form
    return df

def plot_survey_dir(directory):
    for fn in sorted(os.listdir(directory)):
        if fn.lower().endswith('_platesurvey.xml'):
            plot_survey(os.path.join(directory, fn))

def plot_survey(survey):
    if type(survey) == str:
        survey = read_survey(survey)
    form = survey['format'][0]
    gg = plot_plate(survey, form=form, fill='v', discrete=False)
    if 'barcode' in survey.columns:
        gg += p9.ggtitle(survey.iloc[0]['barcode'])
    gg.draw()

def plot_plate(df, form, fill='v', alpha=1, discrete=True, show='target'):
    df = df.copy()
    if form not in [96, 384, 1536]:
        raise ValueError('Only forms 96, 384 and 1536 supported.')
    n_cols = int(sqrt(form/2*3))
    n_rows = int(form / n_cols)
    if show == 'target' and 't_well' in df.columns:
        welln = well2welln(df['t_well'], form)
    elif show == 'target' and 't_welln' in df.columns:
        welln = df['t_welln']
    elif show == 'source' and 's_well' in df.columns:
        welln = well2welln(df['s_well'], form)
    elif show == 'source' and 's_welln' in df.columns:
        welln = df['s_welln']
    else:
        welln = df['welln']
    df['col'] = welln % n_cols + 1
    df['row'] = welln // n_cols + 1
    size = (6, 4) if form < 1536 else (12,8)
    labels = _rows[:n_rows]
    if discrete:
        fill = 'factor(%s)' % fill
    gg = (p9.ggplot(df)
     + p9.aes('col', 'row', fill=fill)
     + p9.geom_tile(p9.aes(width=1, height=1), alpha=alpha)
     + p9.geom_rect(p9.aes(xmin=.5, xmax=n_cols+.5, ymin=.5, ymax=n_rows+.5), color='black', fill=None)
     + p9.scale_x_continuous(limits=(0.5, n_cols+.5), breaks=range(1, n_cols+1)) # minor_breaks=np.arange(1, n_cols+1)+.5
     + p9.scale_y_reverse(limits=(0.5, n_rows+.5), breaks=range(1, n_rows+1), labels=labels)
     + p9.coord_equal(expand=False)
     + p9.theme_minimal()
     + p9.theme(
         figure_size=size,
         axis_ticks=p9.element_blank(),
         axis_title_x=p9.element_blank(),
         axis_title_y=p9.element_blank(),
         panel_grid_major=p9.element_blank(),
         panel_grid_minor=p9.element_line(color='gray'),
         )
    )
    if discrete:
        gg += p9.scale_fill_discrete(l=.4, s=1, color_space='hls')
    else:
        gg += p9.scale_fill_cmap(name='jet', limits=[0, 5])
    return gg

def read_report_csv(fn):
    start = 9
    with open(fn) as file:
        for nn, line in enumerate(file):
            if (nn > start) and (len(line) == 1):
                length = nn - start - 1
                break
    length

    df = pd.read_csv(fn, skiprows=start, nrows=length)
    df = df.rename({
        'SrcWell':'s_well',
        'SrcWellRow':'s_row',
        'SrcWellCol':'s_col',
        'DestWell':'t_well',
        'DestWellRow':'t_row',
        'DestWellCol':'t_col',
        'SkipReason':'skip',
        'FluidComposition':'composition',
        'FluidThickness (mm)':'h_before',
        'VolumeTransferred (nL)':'v',
        'CurrentFluidThickness (mm)':'h_after'}, axis=1)
    # return df
    return df[['s_well',  's_row', 's_col', 't_well', 't_row', 't_col', 'skip', 'composition', 'h_before', 'v', 'h_after']]

def check_log_dir(directory):
    nn, nf = 0, 0
    for fn in sorted(os.listdir(directory)):
        if fn.lower().endswith('_print.xml'):
            nn += 1
            failed = check_log(os.path.join(directory, fn))
            if failed:
                print(fn)
                for fail in failed:
                    nf += 1
                    print('{s_barcode} {s_well} -> {t_barcode}/{t_well} [{v:>4}nl] {reason}'.format(**fail))
    print('%i logs checked, %i failures.' % (nn, nf))

def check_log(fn):
    ll = []
    tree = ET.parse(fn)
    root = tree.getroot()
    s_barcode = root.find("./plateInfo/plate[@type='source']").attrib['barcode']
    t_barcode = root.find("./plateInfo/plate[@type='destination']").attrib['barcode']
    skippedwells = root.find('skippedwells')
    for el in skippedwells:
        ll.append({
            's_barcode': s_barcode,
            's_well': el.attrib['n'],
            't_barcode': t_barcode,
            't_well': el.attrib['dn'],
            'v': el.attrib['vt'],
            'reason': el.attrib['reason'],
        })
    return ll if len(ll) > 0 else None


if __name__ == '__main__':
    pl3 = pd.DataFrame({'t_well':['A1', 'B2', 'H12'], 'v':[1,2,3]})
    plot_picklist(pl3, 96)