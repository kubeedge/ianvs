#!/usr/bin/env python3
"""
Markdown Compiler - Convert MD files to various formats
Supports: HTML, PDF (with weasyprint), and text preview
"""
import markdown
import sys
import os
import argparse
from pathlib import Path

class MarkdownCompiler:
    def __init__(self):
        self.extensions = ['codehilite', 'fenced_code', 'tables', 'toc']
    
    def get_html_template(self, title, content, for_pdf=False):
        """Generate HTML template"""
        if for_pdf:
            # PDF-specific styling
            styles = """
        @page {
            size: A4;
            margin: 2cm;
        }
        body { 
            font-family: 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.5;
            color: #000;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #000;
            page-break-after: avoid;
        }
        code { 
            background-color: #f5f5f5; 
            padding: 2px 4px; 
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre { 
            background-color: #f5f5f5; 
            padding: 10px; 
            border-radius: 3px; 
            border: 1px solid #ddd;
            page-break-inside: avoid;
        }"""
        else:
            # Web-specific styling
            styles = """
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 1.5em;
        }
        code { 
            background-color: #f8f9fa; 
            padding: 2px 6px; 
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9em;
        }
        pre { 
            background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 5px; 
            overflow-x: auto;
            border-left: 4px solid #007acc;
        }
        blockquote {
            border-left: 4px solid #ddd;
            margin: 0;
            padding-left: 1em;
            color: #666;
        }
        a { color: #007acc; text-decoration: none; }
        a:hover { text-decoration: underline; }"""
        
        # Common table styling
        table_styles = """
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        img {
            max-width: 100%;
            height: auto;
        }"""
        
        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        {styles}
        {table_styles}
    </style>
</head>
<body>
{content}
</body>
</html>'''
    
    def md_to_html(self, md_file, output_file=None):
        """Convert markdown to HTML"""
        if not os.path.exists(md_file):
            print(f"❌ Error: {md_file} not found!")
            return False
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html = markdown.markdown(md_content, extensions=self.extensions)
        
        if output_file is None:
            output_file = Path(md_file).with_suffix('.html')
        
        title = Path(md_file).stem
        html_content = self.get_html_template(title, html, for_pdf=False)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML: {md_file} → {output_file}")
        return True
    
    def md_to_pdf(self, md_file, output_file=None):
        """Convert markdown to PDF using weasyprint"""
        try:
            import weasyprint
        except ImportError:
            print("❌ Error: weasyprint is required for PDF conversion")
            print("Install it with: pip install weasyprint")
            return False
        
        if not os.path.exists(md_file):
            print(f"❌ Error: {md_file} not found!")
            return False
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html = markdown.markdown(md_content, extensions=self.extensions)
        
        if output_file is None:
            output_file = Path(md_file).with_suffix('.pdf')
        
        title = Path(md_file).stem
        html_content = self.get_html_template(title, html, for_pdf=True)
        
        weasyprint.HTML(string=html_content).write_pdf(output_file)
        print(f"✅ PDF: {md_file} → {output_file}")
        return True
    
    def md_to_text(self, md_file):
        """Show markdown content as formatted text (preview)"""
        if not os.path.exists(md_file):
            print(f"❌ Error: {md_file} not found!")
            return False
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"\n📄 Preview of {md_file}:")
        print("=" * 60)
        print(content)
        print("=" * 60)
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Markdown Compiler - Convert MD files to various formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python md_compiler.py README.md -f html          # Convert to HTML
  python md_compiler.py README.md -f pdf           # Convert to PDF
  python md_compiler.py README.md -f text          # Preview as text
  python md_compiler.py README.md -f all           # Convert to all formats
  python md_compiler.py README.md -o output.html   # Specify output file
        """
    )
    
    parser.add_argument('input_file', help='Input markdown file')
    parser.add_argument('-f', '--format', 
                       choices=['html', 'pdf', 'text', 'all'], 
                       default='html',
                       help='Output format (default: html)')
    parser.add_argument('-o', '--output', 
                       help='Output file name (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' not found!")
        sys.exit(1)
    
    compiler = MarkdownCompiler()
    
    if args.format == 'html' or args.format == 'all':
        output = args.output if args.output and args.format == 'html' else None
        compiler.md_to_html(args.input_file, output)
    
    if args.format == 'pdf' or args.format == 'all':
        output = args.output if args.output and args.format == 'pdf' else None
        compiler.md_to_pdf(args.input_file, output)
    
    if args.format == 'text':
        compiler.md_to_text(args.input_file)
    
    if args.format == 'all':
        print("\n🎉 All conversions completed!")

if __name__ == "__main__":
    main() 