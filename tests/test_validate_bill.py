import os
import sys
import io
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
from starlette.datastructures import UploadFile


def get_mock_textract_invoice_blocks():
    blocks = [
        {'Id': 'page-1', 'BlockType': 'PAGE'},
        {'Id': 'line-1', 'BlockType': 'LINE', 'Text': 'TAX INVOICE'},
        {'Id': 'line-2', 'BlockType': 'LINE', 'Text': 'GSTIN: 27AABCU9603R1ZM'},
        {'Id': 'line-3', 'BlockType': 'LINE', 'Text': 'Invoice No: INV-2026-001'},
        {'Id': 'line-4', 'BlockType': 'LINE', 'Text': 'Invoice Date: 15-Sep-2026'},

        {
            'Id': 'table-1',
            'BlockType': 'TABLE',
            'Relationships': [{'Type': 'CHILD', 'Ids': [f'cell-{r}-{c}' for r in range(1, 3) for c in range(1, 7)]}]
        },

        {'Id': 'cell-1-1', 'BlockType': 'CELL', 'RowIndex': 1, 'ColumnIndex': 1, 'Relationships': [{'Type': 'CHILD', 'Ids': ['w-h-1']}]},
        {'Id': 'cell-1-2', 'BlockType': 'CELL', 'RowIndex': 1, 'ColumnIndex': 2, 'Relationships': [{'Type': 'CHILD', 'Ids': ['w-h-2']}]},
        {'Id': 'cell-1-3', 'BlockType': 'CELL', 'RowIndex': 1, 'ColumnIndex': 3, 'Relationships': [{'Type': 'CHILD', 'Ids': ['w-h-3']}]},
        {'Id': 'cell-1-4', 'BlockType': 'CELL', 'RowIndex': 1, 'ColumnIndex': 4, 'Relationships': [{'Type': 'CHILD', 'Ids': ['w-h-4']}]},
        {'Id': 'cell-1-5', 'BlockType': 'CELL', 'RowIndex': 1, 'ColumnIndex': 5, 'Relationships': [{'Type': 'CHILD', 'Ids': ['w-h-5']}]},
        {'Id': 'cell-1-6', 'BlockType': 'CELL', 'RowIndex': 1, 'ColumnIndex': 6, 'Relationships': [{'Type': 'CHILD', 'Ids': ['w-h-6']}]},

        {'Id': 'w-h-1', 'BlockType': 'WORD', 'Text': 'Item Description'},
        {'Id': 'w-h-2', 'BlockType': 'WORD', 'Text': 'HSN/SAC'},
        {'Id': 'w-h-3', 'BlockType': 'WORD', 'Text': 'Qty'},
        {'Id': 'w-h-4', 'BlockType': 'WORD', 'Text': 'Unit Price'},
        {'Id': 'w-h-5', 'BlockType': 'WORD', 'Text': 'Taxable Amount'},
        {'Id': 'w-h-6', 'BlockType': 'WORD', 'Text': 'Total Amount'},

        {'Id': 'cell-2-1', 'BlockType': 'CELL', 'RowIndex': 2, 'ColumnIndex': 1, 'Relationships': [{'Type': 'CHILD', 'Ids': ['w-d-1']}]},
        {'Id': 'cell-2-2', 'BlockType': 'CELL', 'RowIndex': 2, 'ColumnIndex': 2, 'Relationships': [{'Type': 'CHILD', 'Ids': ['w-d-2']}]},
        {'Id': 'cell-2-3', 'BlockType': 'CELL', 'RowIndex': 2, 'ColumnIndex': 3, 'Relationships': [{'Type': 'CHILD', 'Ids': ['w-d-3']}]},
        {'Id': 'cell-2-4', 'BlockType': 'CELL', 'RowIndex': 2, 'ColumnIndex': 4, 'Relationships': [{'Type': 'CHILD', 'Ids': ['w-d-4']}]},
        {'Id': 'cell-2-5', 'BlockType': 'CELL', 'RowIndex': 2, 'ColumnIndex': 5, 'Relationships': [{'Type': 'CHILD', 'Ids': ['w-d-5']}]},
        {'Id': 'cell-2-6', 'BlockType': 'CELL', 'RowIndex': 2, 'ColumnIndex': 6, 'Relationships': [{'Type': 'CHILD', 'Ids': ['w-d-6']}]},

        {'Id': 'w-d-1', 'BlockType': 'WORD', 'Text': 'Legal Consulting Services'},
        {'Id': 'w-d-2', 'BlockType': 'WORD', 'Text': '998311'},
        {'Id': 'w-d-3', 'BlockType': 'WORD', 'Text': '1'},
        {'Id': 'w-d-4', 'BlockType': 'WORD', 'Text': '10000.00'},
        {'Id': 'w-d-5', 'BlockType': 'WORD', 'Text': '10000.00'},
        {'Id': 'w-d-6', 'BlockType': 'WORD', 'Text': '11800.00'},

        {'Id': 'line-5', 'BlockType': 'LINE', 'Text': 'CGST @ 9%: INR 900.00'},
        {'Id': 'line-6', 'BlockType': 'LINE', 'Text': 'SGST @ 9%: INR 900.00'},
        {'Id': 'line-7', 'BlockType': 'LINE', 'Text': 'Total GST @ 18%: INR 1800.00'},
        {'Id': 'line-8', 'BlockType': 'LINE', 'Text': 'Grand Total: INR 11800.00'}
    ]
    return blocks


async def test_bill_validation():
    print('\n--- 1. Testing Textract Parsing & Unstructured Formatting ---')
    blocks = get_mock_textract_invoice_blocks()
    tables, line_items, raw_lines = main.parse_textract_tables_and_lines(blocks)

    assert len(tables) == 1, 'Should have extracted 1 table'
    assert len(line_items) == 1, 'Should have extracted 1 line item'
    assert line_items[0]['hsn_sac'] == '998311'
    assert line_items[0]['unit_price'] == 10000.0

    formatted_text, structured_tables = main.format_textract_with_unstructured(tables, raw_lines)
    print('\n--- Unstructured Formatted Text Preview ---')
    print(formatted_text[:300])
    assert structured_tables[0].html is not None, 'HTML representation missing'
    assert 'Legal Consulting Services' in formatted_text

    tax_summary = main.extract_tax_rates_summary(tables, line_items, formatted_text)
    print('\n--- Extracted Tax Rates Summary ---')
    print(f'Detected GST Rates: {tax_summary.detected_tax_rates}')
    print(f'CGST Rates: {tax_summary.cgst_rates}')
    print(f'SGST Rates: {tax_summary.sgst_rates}')

    assert '18%' in tax_summary.detected_tax_rates, '18% GST should be detected'
    assert '9%' in tax_summary.cgst_rates, '9% CGST should be detected'

    print('\n--- 2. Testing POST /validate-bill Endpoint with S3 & Textract Mock ---')
    mock_s3 = MagicMock()
    mock_textract = MagicMock()
    mock_textract.analyze_document.return_value = {'Blocks': blocks}

    with patch('boto3.client') as mock_boto:
        def client_side_effect(service_name, **kwargs):
            if service_name == 's3':
                return mock_s3
            elif service_name == 'textract':
                return mock_textract
            return MagicMock()

        mock_boto.side_effect = client_side_effect

        dummy_content = b'%PDF-1.4 test invoice content'
        upload_file = UploadFile(
            file=io.BytesIO(dummy_content),
            filename='sample_gst_invoice.pdf',
            headers={'content-type': 'application/pdf'}
        )

        response = await main.validate_bill(file=upload_file)

        print(f'Response status: {response.status}')
        print(f'S3 Bucket: {response.s3_bucket}')
        print(f'S3 Key: {response.s3_key}')
        print(f'Detected Tax Rates: {response.tax_rates.detected_tax_rates}')
        print(f'Line items count: {len(response.line_items)}')

        mock_s3.put_object.assert_called_once()
        uploaded_bucket = mock_s3.put_object.call_args[1]['Bucket']
        expected_bucket = os.getenv("S3_BUCKET_NAME", "gst-rag-invoices-slash-020")
        assert uploaded_bucket == expected_bucket, f'Wrong bucket: {uploaded_bucket}'

        mock_textract.analyze_document.assert_called_once()
        textract_doc = mock_textract.analyze_document.call_args[1]['Document']['S3Object']
        assert textract_doc['Bucket'] == expected_bucket

        mock_s3.delete_object.assert_called_once()
        deleted_bucket = mock_s3.delete_object.call_args[1]['Bucket']
        assert deleted_bucket == expected_bucket, f'Wrong deleted bucket: {deleted_bucket}'
        print(f'S3 temporary object deletion confirmed in finally block for bucket: {deleted_bucket}')

        # Verify AWS region configuration passed to boto3 clients
        expected_region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-south-1"))
        for call_args in mock_boto.call_args_list:
            region_arg = call_args[1].get('region_name')
            assert region_arg == expected_region, f'Expected region {expected_region}, got {region_arg}'
        print(f'AWS clients verified with region: {expected_region}')

        assert response.status == 'success'
        assert len(response.tax_rates.detected_tax_rates) > 0
        assert len(response.line_items) > 0

    print('\n--- 3. Testing Validation Error Handling ---')
    txt_file = UploadFile(
        file=io.BytesIO(b'hello'),
        filename='invoice.txt',
        headers={'content-type': 'text/plain'}
    )
    try:
        await main.validate_bill(file=txt_file)
        assert False, 'Should have rejected .txt file'
    except main.HTTPException as e:
        print(f'Unsupported extension rejected: status {e.status_code} - {e.detail}')
        assert e.status_code == 400

    print('\nAll /validate-bill tests passed successfully!')

if __name__ == '__main__':
    asyncio.run(test_bill_validation())