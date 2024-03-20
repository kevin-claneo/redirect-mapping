import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from huggingface_hub import HfFolder
import time

def main():
    st.title("Site Migration Redirects Automator v.1")
    page_icon=":arrow_right_hook:",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/kirchhoff-kevin/',
        'About': "This is an app for checking your topical authority! Adapted from Lee Foot's GSC-connector check out his apps: https://leefoot.co.uk"
    }
    )
    st.image("https://www.claneo.com/wp-content/uploads/Element-4.svg", width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.caption(":point_right: Join Claneo and support exciting clients as part of the Consulting team") 
    st.caption(':bulb: Make sure to mention that *Kevin* brought this job posting to your attention')
    st.link_button("Learn More", "https://www.claneo.com/en/career/#:~:text=Consulting")
    st.markdown("""
    ##### Before Using the Tool
    
    To ensure the effectiveness of this tool in mapping redirects, it is essential to adequately prepare the input data. This process begins with exporting data from *Screaming Frog*.
    
    ##### 👉🏼 Prepare Data with Screaming Frog
    
    1. Run a full crawl of your website using Screaming Frog.
    2. Filter the crawl results to include only HTML pages with a status code of 200, ensuring to remove duplicate or unnecessary URLs for redirect mapping.
    3. Export the filtered results to a CSV file. Ensure the file contains columns for the URL address, title, meta description, and other relevant information you wish to use for matching.
    4. Repeat the process for the destination website, running a crawl of the site in staging (or the new site) and exporting the results.
    
    ##### 👉🏼 Instructions
    
    1. Prepare the CSV files containing the URLs of the original site (`origin.csv`) and the destination site (`destination.csv`) following the instructions above.
    2. Upload the CSV files using the provided uploaders.
    3. Select the relevant columns for matching from the dropdown menu.
    4. Click the "Match URLs" button to start the matching process.
    5. Download the results directly from the interface, which will include the origin URLs, the corresponding destination URLs, and the similarity matching.
    
    ###### Credits
    
    This tool is based on the original Python script [Automated Redirect Matchmaker for Site Migrations](https://colab.research.google.com/drive/1Y4msGtQf44IRzCotz8KMy0oawwZ2yIbT?usp=sharing) developed by [Daniel Emery](https://www.linkedin.com/in/dpe1/), which provides an automated approach to mapping redirects during website migrations. The tool has been extended and integrated into a Streamlit application to improve interactivity and usability.
    """)
    
    st.markdown("---")
    
    # Field for entering the Hugging Face token
    hf_token = str(st.secrets["installed"]["hf_token"])
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from huggingface_hub import HfFolder
import time
from stqdm import stqdm # Import stqdm

def main():
    st.title("Site Migration Redirects Automator v.1")
    st.markdown("""
    ##### Before Using the Tool
    
    To ensure the effectiveness of this tool in mapping redirects, it is essential to adequately prepare the input data. This process begins with exporting data from *Screaming Frog*.
    
    ##### 👉🏼 Prepare Data with Screaming Frog
    
    1. Run a full crawl of your website using Screaming Frog.
    2. Filter the crawl results to include only HTML pages with a status code of 200, ensuring to remove duplicate or unnecessary URLs for redirect mapping.
    3. Export the filtered results to a CSV file. Ensure the file contains columns for the URL address, title, meta description, and other relevant information you wish to use for matching.
    4. Repeat the process for the destination website, running a crawl of the site in staging (or the new site) and exporting the results.
    
    ##### 👉🏼 Instructions
    
    1. Prepare the CSV files containing the URLs of the original site (`origin.csv`) and the destination site (`destination.csv`) following the instructions above.
    2. Upload the CSV files using the provided uploaders.
    3. Select the relevant columns for matching from the dropdown menu.
    4. Click the "Match URLs" button to start the matching process.
    5. Download the results directly from the interface, which will include the origin URLs, the corresponding destination URLs, and the similarity matching.
    
    ###### Credits
    
    This tool is based on the original Python script [Automated Redirect Matchmaker for Site Migrations](https://colab.research.google.com/drive/1Y4msGtQf44IRzCotz8KMy0oawwZ2yIbT?usp=sharing) developed by [Daniel Emery](https://www.linkedin.com/in/dpe1/), which provides an automated approach to mapping redirects during website migrations. The tool has been extended and integrated into a Streamlit application to improve interactivity and usability.
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
            origin_embeddings = model.encode(origin_df['combined_text'].tolist(), show_progress_bar=True)
            destination_embeddings = model.encode(destination_df['combined_text'].tolist(), show_progress_bar=True)

            dimension = origin_embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)
            faiss_index.add(destination_embeddings.astype('float32'))

            # Initialize progress bar
            progress_bar = st.progress(0)

            # Use stqdm to wrap the loop for real-time progress updates
            for i in stqdm(range(len(origin_embeddings))):
                distances, indices = faiss_index.search(origin_embeddings[i].reshape(1, -1).astype('float32'), k=1)
                similarity_scores = 1 - (distances / np.max(distances))
                # Update your data structures here based on the results
                # This is just a placeholder for where you would update your results

            # Creation of series to handle different lengths
            matched_url_series = pd.Series(destination_df['Address'].iloc[indices.flatten()].values, index=origin_df.index)
            similarity_scores_series = pd.Series(similarity_scores.flatten(), index=origin_df.index)

            # Creation of the results DataFrame
            results_df = pd.DataFrame({
                'origin_url': origin_df['Address'],
                'matched_url': matched_url_series,
                'similarity_score': similarity_scores_series
            })

            # Convert DataFrame to CSV string
            csv_string = results_df.to_csv(index=False)

            # Display download button
            if st.button("Download Results"):
                st.download_button(
                    label="Download Results",
                    data=csv_string,
                    file_name='redirect_mapping_results.csv',
                    mime='text/csv'
                )

if __name__ == "__main__":
    main()
