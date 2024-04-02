import streamlit as st
from streamlit_elements import Elements
import base64
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from huggingface_hub import HfFolder
from stqdm import stqdm

st.set_page_config(
    page_title="AI Driven Redirect Mapping",
    page_icon=":arrow_right_hook:",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/kirchhoff-kevin/',
        'About': "This is an app for finding the matching redirect URLs using the FAISS model."
    }
)

def show_dataframe(report):
    """
    Shows a preview of the first 100 rows of the report DataFrame in an expandable section.
    """
    with st.expander("Preview the First 100 Rows"):
        st.dataframe(report.head(DF_PREVIEW_ROWS))

def download_csv_link(report):
    """
    Generates and displays a download link for the report DataFrame in CSV format.
    """
    def to_csv(df):
        return df.to_csv(index=False, encoding='utf-8-sig')

    csv = to_csv(report)
    b64_csv = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64_csv}" download="redirect_mapping_results.csv</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    st.image("https://www.claneo.com/wp-content/uploads/Element-4.svg", width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.caption(":point_right: Join Claneo and support exciting clients as part of the Consulting team") 
    st.caption(':bulb: Make sure to mention that *Kevin* brought this job posting to your attention')
    st.link_button("Learn More", "https://www.claneo.com/en/career/#:~:text=Consulting")
    st.title("Use Facebook AI Similarity Search (FAISS) to find matching redirect URLs")
    st.caption('Adapted by [Kevin](https://www.linkedin.com/in/kirchhoff-kevin/)') 
    st.markdown("""
    This tool is based on the original Python script [Automated Redirect Matchmaker for Site Migrations](https://colab.research.google.com/drive/1Y4msGtQf44IRzCotz8KMy0oawwZ2yIbT?usp=sharing) developed by [Daniel Emery](https://www.linkedin.com/in/dpe1/).
    
    ## Before Using the Tool 
    To ensure the effectiveness of this tool in mapping redirects, it is essential to adequately prepare the input data. This process begins with exporting data from *Screaming Frog*.
   
    #### 🐸 Data Preparation with Screaming Frog
    
    1. Run a full crawl of your website using Screaming Frog.
    2. Filter the crawl results to include only HTML pages with a status code of 200, ensuring to remove duplicate or unnecessary URLs for redirect mapping.
    3. Export the filtered results to a CSV file. Ensure the file contains columns for the URL address, title, meta description, and other relevant information you wish to use for matching.
    4. Repeat the process for the destination website, running a crawl of the site in staging (or the new site) and exporting the results.
    
    ###  ⚠️ Instructions for this Tool
    
    1. Prepare the CSV files containing the URLs of the original site (`origin.csv`) and the destination site (`destination.csv`) following the instructions above.
    2. Upload the CSV files using the provided uploaders.
    3. Select the relevant columns for matching from the dropdown menu.
    4. Click the "Match URLs" button to start the matching process.
    5. Download the results via "Download Results".
    """)
    
    st.markdown("---")

    # Field for entering the Hugging Face token
    hf_token = str(st.secrets["installed"]["hf_token"])
    
    if hf_token:
        # Set the Hugging Face token
        HfFolder.save_token(hf_token)

    # Loading CSV files
    origin_file = st.file_uploader("Upload the origin.csv file", type="csv")
    destination_file = st.file_uploader("Upload the destination.csv file", type="csv")

    if origin_file and destination_file:
        origin_df = pd.read_csv(origin_file)
        destination_df = pd.read_csv(destination_file)
    
        # Identification of common columns
        common_columns = list(set(origin_df.columns) & set(destination_df.columns))
        selected_columns = st.multiselect("Select columns to use for similarity matching:", common_columns)
    
        if st.button("Match URLs") and selected_columns:
            # Preprocessing of data
            origin_df['combined_text'] = origin_df[selected_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            destination_df['combined_text'] = destination_df[selected_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
            # Matching of data
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            progress_bar_origin =  None
            
            # Use stqdm to wrap the loop for real-time progress updates for origin texts
            for i in stqdm(range(len(origin_df)), mininterval=0.5, desc="Encoding origin texts"):
                origin_embeddings = model.encode(origin_df['combined_text'].iloc[i:i+1].tolist(), show_progress_bar=False)
                progress_value = (i + 1) / len(origin_df)
                if progress_bar_origin is not None:
                    progress_bar_origin.progress(progress_value)
                else:
                    progress_bar_origin = st.progress(0.0)


            progress_value  = 0
            progress_bar_destination = None
            
            # Use stqdm to wrap the loop for real-time progress updates for destination texts
            for i in stqdm(range(len(destination_df)), mininterval=0.5, desc="Encoding destination texts"):
                destination_embeddings = model.encode(destination_df['combined_text'].iloc[i:i+1].tolist(), show_progress_bar=False)
                progress_value = (i + 1) / len(destination_df)
                progress_bar_destination.progress(progress_value)
                if progress_bar_destination is not None:
                    progress_bar_destination.progress(progress_value)
                else:
                    progress_bar_destination = st.progress(0.0)
            
            # After encoding and before creating the results DataFrame
            
            # Create a FAISS index for the destination embeddings
            dimension = origin_embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)
            faiss_index.add(destination_embeddings.astype('float32'))
            
            # Search for the nearest neighbors
            distances, indices = faiss_index.search(origin_embeddings.astype('float32'), k=1)
            similarity_scores = 1 - (distances / np.max(distances))
            
            # Creation of series to handle different lengths
            matched_url_series = pd.Series(destination_df['Address'].iloc[indices.flatten()].values, index=origin_df.index)
            similarity_scores_series = pd.Series(similarity_scores.flatten(), index=origin_df.index)
            
            # Creation of the results DataFrame
            report = pd.DataFrame({
                'origin_url': origin_df['Address'],
                'matched_url': matched_url_series,
                'similarity_score': similarity_scores_series
            })
            
            if report is not None: 
                show_dataframe(report)
                st.write(len(report))
                download_csv_link(report)

if __name__ == "__main__":
    main()
