# Skill: Advanced Web Research via Jina Reader (r.jina.ai)

## 1. Core Function
Use Jina Reader to convert complex URLs, PDFs, and JS-heavy sites (like Notion, LinkedIn, Twitter) into clean, LLM-friendly Markdown.

## 2. Usage Patterns
- **Basic**: Prepend `https://r.jina.ai/` to any URL.
- **Search**: Use `https://s.jina.ai/?q=YOUR_QUERY` to get a summarized web search result.
- **PDF**: Prepend the prefix to a direct PDF link for high-quality text extraction.

## 3. Optimization Headers (for tool calls)
When making a web request, consider these headers to save tokens and improve quality:
- `X-Target-Selector`: Specify a CSS selector (e.g., `article`, `.main-content`) to retrieve ONLY the relevant part.
- `X-With-Generated-Alt`: Set to `true` to get AI-generated descriptions for images.
- `X-With-Links-Summary`: Set to `true` to get a list of all links at the end of the document (useful for multi-step crawling).
- `X-Return-Format`: Default is `markdown`. Use `text` for even more token savings if formatting isn't needed.

## 4. When to use Jina instead of standard tools:
1. The site is a Single Page Application (SPA) or requires JavaScript.
2. The site is behind a soft paywall or has aggressive anti-bot measures.
3. The page is extremely long/noisy (Jina's cleaning is superior).
4. You need to "read" a PDF from a URL.
5. You need to see descriptions of images/charts on the page.

## 5. Implementation Note
If a standard `read_web_page` call fails or returns "JS required" / "Captcha", immediately retry using the `https://r.jina.ai/` prefix.
