from automas.mcp.servers.document.docx_reader import docx_server
from automas.mcp.servers.document.pdf_reader import pdf_server
from automas.mcp.servers.document.pptx_reader import pptx_server
from automas.mcp.servers.document.xlsx_reader import xlsx_server
from automas.mcp.servers.document.zip_extractor import zip_server

__all__ = [
    "pdf_server",
    "docx_server",
    "pptx_server",
    "xlsx_server",
    "zip_server",
]
