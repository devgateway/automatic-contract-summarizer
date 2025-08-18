from bs4 import BeautifulSoup, Tag
import html


def simplify_html(html_):
    soup = BeautifulSoup(html_, 'html.parser')

    # Remove Non-Structural Tags
    non_structural_tags = [
        'script', 'style', 'meta', 'link', 'noscript', 'iframe',
        'embed', 'object', 'param', 'source', 'track', 'wbr'
    ]
    for tag in soup(non_structural_tags):
        tag.decompose()

    # Strip Unnecessary Attributes
    for tag in soup.find_all():
        tag.attrs = {}

    # Simplify Tables
    def simplify_table(table_tag):
        simplified_rows = []
        for tr in table_tag.find_all('tr'):
            cells = tr.find_all(['td', 'th'])
            cell_texts = [' '.join(cell.stripped_strings) for cell in cells]
            simplified_row = '||'.join(cell_texts)
            if simplified_row:
                simplified_rows.append('<row>')
                simplified_rows.append(simplified_row)
                simplified_rows.append('</row>')
        simplified_table = '\n'.join(simplified_rows)
        # Replace the table tag with the simplified text
        table_tag.replace_with(simplified_table)

    # Simplify Tags Recursively
    def simplify_tag(tag):
        # Remove <p> tags by replacing them with their contents
        if tag.name == 'p':
            contents = tag.contents
            tag.unwrap()  # Remove the <p> tag but keep its contents
            for content in contents:
                if isinstance(content, Tag):
                    simplify_tag(content)
            return

        # Simplify <table> tags
        if tag.name == 'table':
            simplify_table(tag)
            return

        # Flatten nested structures when possible
        while len(tag.contents) == 1 and isinstance(tag.contents[0], Tag):
            child = tag.contents[0]
            if not tag.attrs and not child.attrs and tag.name == child.name:
                tag.replace_with(child)
                tag = child
            else:
                break

        # Remove redundant div and span tags without attributes
        if tag.name in ['div', 'span'] and not tag.attrs:
            children = []
            for child in tag.contents:
                if isinstance(child, Tag):
                    simplify_tag(child)
                children.append(child)
            tag.replace_with(*children)
        else:
            for child in tag.contents:
                if isinstance(child, Tag):
                    simplify_tag(child)

    #simplify_tag(soup)

    # Use Shorter Tag Names or Symbols
    tag_name_mapping = {
        'section': 's',
        'article': 'a',
        'header': 'h',
        'head': 'hd',
        'title': 'tt',
        'footer': 'f',
        'nav': 'n',
        'main': 'm',
        'aside': 'as',
        'div': 'd',
        'span': 'sp',
        'paragraph': 'p',  # Note: 'paragraph' is not an HTML tag, included for completeness
        'heading': 'h',
        'ul': 'u',
        'ol': 'o',
        'li': 'l',
        'table': 't',
        'thead': 'th',
        'tbody': 'tb',
        'tr': 'r',
        'td': 'd',
        'th': 'h',
        'a': 'a',
        'img': 'i',
        'strong': 'st',
        'em': 'e',
        'b': 'b',
        'i': 'i',
        'u': 'u',
        'small': 'sm',
        'big': 'bg',
        'blockquote': 'bq',
        'pre': 'pr',
        'code': 'c',
        'cite': 'ci',
        'body': 'bd',
        'html': 'ht',
        # Add any other tags you wish to shorten
    }

    for tag in soup.find_all():
        original_name = tag.name
        if tag.name in tag_name_mapping:
            tag.name = tag_name_mapping[tag.name]

    # Minify the HTML Code
    simplified_html = ' '.join(str(soup).split())
    simplified_html = (simplified_html.replace('> <', '><')
                       .replace('> ', '>')
                       .replace(' <', '<')
                       .replace('<a>', ' ')
                       .replace('</a>', ' '))

    return html.unescape(simplified_html)
