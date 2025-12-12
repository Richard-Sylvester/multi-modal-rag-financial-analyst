import os
from unstructured.partition.pdf import partition_pdf

def load_pdf_documents(file_path):
    """
    Splits the PDF into 'Text' elements and 'Table' elements.
    """
    print(f"🚀 Analyzing PDF structure: {file_path}...")
    
    # This function uses AI vision to detect tables vs text
    raw_pdf_elements = partition_pdf(
        filename=file_path,
        infer_table_structure=True,        # Essential for reading financial tables
        chunking_strategy="by_title",      # Groups text logically (e.g. by section header)
        max_characters=4000,               # Don't make chunks too big
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )
    
    # Buckets for our sorted data
    text_elements = []
    table_elements = []
    
    # Sorting loop
    for element in raw_pdf_elements:
        if "Table" in str(type(element)):
            table_elements.append(element)
        elif "CompositeElement" in str(type(element)):
            text_elements.append(element)

    print(f"✅ Extraction Done: {len(text_elements)} text blocks, {len(table_elements)} tables.")
    return text_elements, table_elements