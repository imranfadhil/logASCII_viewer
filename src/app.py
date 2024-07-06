import streamlit as st
import logASCII_viewer.utils.LAS_handler as handler
import logASCII_viewer.utils.LAS_visualization as vis

if __name__ == '__main__':
    unit = 'imperial'
    st.sidebar.subheader('Import LAS files')
    las_file = st.sidebar.file_uploader('Upload LAS2.0 file', type=['LAS', 'las'], accept_multiple_files=True)
    if las_file:
        dfa, header = handler.read_las_file(las_file, unit)

        wells = dfa.WELLNAME.unique()
        st.sidebar.subheader('Select a Well')
        well = st.sidebar.selectbox('', wells)

        if well:
            df = dfa[dfa.WELLNAME == well]

            st.sidebar.subheader('Select the Raw Logs')
            log_groups = ['GR', ('RT', 'RES', 'LL'), ('RHOB', 'DEN'), ('NEU', 'NPH')]
            logs = df.columns
            gr = st.sidebar.selectbox('Select Gamma Ray', [x for x in logs if x.startswith(log_groups[0])])
            res = st.sidebar.selectbox('Select Resistivity', [x for x in logs if x.startswith(log_groups[1])])
            rhob = st.sidebar.selectbox('Select Bulk Density', [x for x in logs if x.startswith(log_groups[2])])
            nphi = st.sidebar.selectbox('Select Neutron', [x for x in logs if x.startswith(log_groups[3])])
            rawLog = [gr, res, rhob, nphi]

            with st.expander('Well Header Information'):
                st.write(header.head(10))

            with st.expander('Well Curve Information'):
                st.write(df.describe().transpose())

            vis.plotly_log(df, rawLog, unit)
