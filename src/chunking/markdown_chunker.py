"""Split parsed Markdown into hierarchical, token-bounded chunks."""

from __future__ import annotations

from markdown_it.token import Token

from src.preprocessing.normalize import ParsedMarkdown
from src.preprocessing.tokenizer import count_tokens


def chunk_markdown(
    markdown: ParsedMarkdown,
    max_tokens: int,
    overlap_tokens: int,
) -> list[dict]:
    """Split a parsed Markdown document into retrieval chunks.

    First, the document is segmented into Markdown sections.
    Sections that fit within the token limit remain as a single chunk.
    Larger sections are split into multiple chunks with the configured
    overlap.

    Args:
        markdown: Normalized Markdown text and its parsed Markdown tokens.
        max_tokens: Maximum target number of tokens per chunk.
        overlap_tokens: Number of tokens to reuse between consecutive chunks.

    Returns:
        A list of dictionaries representing the generated chunks.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")

    if overlap_tokens >= max_tokens and overlap_tokens > 0:
        raise ValueError("overlap_tokens must be less than max_tokens.")

    sections = _split_sections(markdown)
    chunks = []

    for section in sections:
        if section["token_count"] <= max_tokens:
            chunks.append(section)
            continue

        chunks.extend(
            _split_large_section(
                section,
                max_tokens,
                overlap_tokens,
            )
        )

    return chunks


def _split_sections(
    markdown: ParsedMarkdown,
) -> list[dict]:
    """Split a Markdown document into sections based on headings.

    Markdown headings are obtained from the tokens produced by
    ``markdown-it-py``. The function also builds the hierarchical section
    path, such as ``Security > OAuth2 > JWT Tokens``.

    Args:
        markdown: Normalized Markdown text and parsed tokens.

    Returns:
        A list of section dictionaries containing their content,
        hierarchy, title, and token count.
    """
    tokens = markdown.tokens
    lines = markdown.text.splitlines()

    sections = []
    heading_stack: list[str] = []

    section_start_line = 0
    section = ""
    title = ""

    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue

        if token.map is None:
            continue

        heading_title = _get_heading_title(tokens, index)

        if heading_title is None:
            continue

        heading_level = int(token.tag[1:])
        heading_start_line = token.map[0]

        content = _extract_lines(
            lines,
            section_start_line,
            heading_start_line,
        )

        if content.strip():
            sections.append(
                _create_section(
                    content=content,
                    section=section,
                    title=title,
                )
            )

        heading_stack = heading_stack[: heading_level - 1]
        heading_stack.append(heading_title)

        section = " > ".join(heading_stack)
        title = heading_title
        section_start_line = heading_start_line

    content = _extract_lines(
        lines,
        section_start_line,
        len(lines),
    )

    if content.strip():
        sections.append(
            _create_section(
                content=content,
                section=section,
                title=title,
            )
        )

    return sections


def _get_heading_title(
    tokens: list[Token],
    heading_index: int,
) -> str | None:
    """Extract the title of a heading from Markdown parser tokens.

    A Markdown heading is represented by an ``heading_open`` token followed
    by an ``inline`` token containing the heading text.

    Args:
        tokens: Tokens produced by ``markdown-it-py``.
        heading_index: Index of the heading opening token.

    Returns:
        The heading text, or ``None`` when the expected inline token is not
        present.
    """
    if heading_index + 1 >= len(tokens):
        return None

    inline_token = tokens[heading_index + 1]

    if inline_token.type != "inline":
        return None

    return inline_token.content


def _extract_lines(
    lines: list[str],
    start_line: int,
    end_line: int,
) -> str:
    """Extract and join a range of Markdown lines.

    Args:
        lines: Lines of the normalized Markdown document.
        start_line: Inclusive starting line index.
        end_line: Exclusive ending line index.

    Returns:
        The extracted lines joined into a single string.
    """
    return "\n".join(
        lines[start_line:end_line]
    ).strip()


def _create_section(
    content: str,
    section: str,
    title: str,
) -> dict:
    """Create the internal representation of a Markdown section.

    Args:
        content: Section content.
        section: Full hierarchical section path.
        title: Title of the current section.

    Returns:
        A dictionary containing section metadata and token count.
    """
    return {
        "section": section,
        "title": title,
        "content": content,
        "token_count": count_tokens(content),
    }


def _split_large_section(
    section: dict,
    max_tokens: int,
    overlap_tokens: int,
) -> list[dict]:
    """Split a section that exceeds the configured token limit.

    The section is divided into Markdown blocks instead of arbitrarily
    cutting lines. Blocks that fit together within ``max_tokens`` are
    grouped into the same chunk.

    Oversized text blocks are split by words. Fenced code blocks are split
    by source lines and each generated chunk receives its own fence.

    Args:
        section: Section to split.
        max_tokens: Maximum target number of tokens per chunk.
        overlap_tokens: Desired overlap between consecutive chunks.

    Returns:
        A list of chunks generated from the large section.
    """
    blocks = _get_blocks(section["content"])
    chunks = []
    current_blocks: list[str] = []

    for block in blocks:
        block_tokens = count_tokens(block)

        if block_tokens > max_tokens:
            if current_blocks:
                chunks.append(
                    _create_chunk(
                        section,
                        current_blocks,
                    )
                )

                current_blocks = _get_overlap_blocks(
                    current_blocks,
                    overlap_tokens,
                )

            if _is_code_block(block):
                if current_blocks:
                    chunks.append(_create_chunk(section, current_blocks))
                    current_blocks = []
                chunks.extend(
                    _split_code_block(
                        section,
                        block,
                        max_tokens,
                    )
                )
            else:
                split_chunks = _split_text_block(
                    section,
                    block,
                    max_tokens,
                )
                if current_blocks and split_chunks:
                    first_content = split_chunks[0]["content"]
                    combined = "\n\n".join(
                        current_blocks + [first_content]
                    )
                    if count_tokens(combined) <= max_tokens:
                        split_chunks[0] = _create_chunk(
                            section,
                            [combined],
                        )
                    else:
                        chunks.append(
                            _create_chunk(section, current_blocks)
                        )
                    current_blocks = []
                chunks.extend(split_chunks)

            current_blocks = []
            continue

        if _would_exceed_limit(
            current_blocks,
            block,
            max_tokens,
        ):
            chunks.append(
                _create_chunk(
                    section,
                    current_blocks,
                )
            )

            current_blocks = _get_overlap_blocks(
                current_blocks,
                overlap_tokens,
            )
            if _would_exceed_limit(
                current_blocks,
                block,
                max_tokens,
            ):
                current_blocks = []

        current_blocks.append(block)

    if current_blocks:
        chunks.append(
            _create_chunk(
                section,
                current_blocks,
            )
        )

    return chunks


def _get_blocks(content: str) -> list[str]:
    """Split section content into natural Markdown blocks.

    Blank lines are used as block boundaries so paragraphs, lists, code
    blocks, and other Markdown structures remain grouped together instead
    of being split line by line.

    Args:
        content: Markdown section content.

    Returns:
        A list of non-empty Markdown blocks.
    """
    blocks = []
    current_lines = []

    in_fence = False

    for line in content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            current_lines.append(line)
            continue

        if line.strip():
            current_lines.append(line)
            continue

        if current_lines and not in_fence:
            blocks.append(
                "\n".join(current_lines)
            )
            current_lines = []

    if current_lines:
        blocks.append(
            "\n".join(current_lines)
        )

    return blocks


def _is_code_block(block: str) -> bool:
    """Return whether a block is a fenced Markdown code block."""
    lines = block.splitlines()
    if len(lines) < 2:
        return False
    opening = lines[0].lstrip()
    closing = lines[-1].lstrip()
    return (
        (opening.startswith("```") and closing.startswith("```"))
        or (opening.startswith("~~~") and closing.startswith("~~~"))
    )


def _split_text_block(
    section: dict,
    block: str,
    max_tokens: int,
) -> list[dict]:
    """Split an oversized non-code block without exceeding the limit."""
    chunks: list[dict] = []
    words: list[str] = []
    current: list[str] = []

    for word in block.split():
        candidate = " ".join(current + [word])
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(_create_chunk(section, [" ".join(current)]))
            current = []
        current.append(word)

    if current:
        words.append(" ".join(current))

    chunks.extend(
        _create_chunk(section, [text])
        for text in words
    )
    return chunks


def _split_code_block(
    section: dict,
    block: str,
    max_tokens: int,
) -> list[dict]:
    """Split an oversized fenced code block without losing Markdown fences."""
    lines = block.splitlines()
    opening = lines[0]
    closing = lines[-1]
    code_lines = lines[1:-1]
    fence_tokens = count_tokens(f"{opening}\n{closing}")
    content_limit = max_tokens - fence_tokens
    if content_limit <= 0:
        raise ValueError(
            "max_tokens is too small to preserve the code block fences."
        )

    chunks: list[dict] = []
    current_lines: list[str] = []

    for line in code_lines:
        candidate_lines = current_lines + [line]
        candidate = "\n".join(
            [opening, *candidate_lines, closing]
        )

        if current_lines and count_tokens(candidate) > max_tokens:
            chunks.append(
                _create_chunk(
                    section,
                    ["\n".join([opening, *current_lines, closing])],
                )
            )
            current_lines = [line]
            continue

        if not current_lines and count_tokens(candidate) > max_tokens:
            line_chunks = _split_text_block(
                section,
                line,
                content_limit,
            )
            chunks.extend(
                _create_chunk(
                    section,
                    [
                        "\n".join(
                            [opening, part["content"], closing]
                        )
                    ],
                )
                for part in line_chunks
            )
            continue

        current_lines.append(line)

    if current_lines:
        chunks.append(
            _create_chunk(
                section,
                ["\n".join([opening, *current_lines, closing])],
            )
        )

    return chunks


def _would_exceed_limit(
    blocks: list[str],
    block: str,
    max_tokens: int,
) -> bool:
    """Check whether adding a block would exceed the token limit.

    Args:
        blocks: Blocks currently assigned to a chunk.
        block: Candidate block to add.
        max_tokens: Maximum target number of tokens.

    Returns:
        ``True`` when the combined content exceeds ``max_tokens``.
    """
    if not blocks:
        return False

    content = "\n\n".join(
        blocks + [block]
    )

    return count_tokens(content) > max_tokens


def _create_chunk(
    section: dict,
    blocks: list[str],
) -> dict:
    """Create a chunk from a collection of Markdown blocks.

    Args:
        section: Source section metadata.
        blocks: Markdown blocks included in the chunk.

    Returns:
        A dictionary containing the chunk content and metadata.
    """
    content = "\n\n".join(blocks).strip()

    return {
        "section": section["section"],
        "title": section["title"],
        "content": content,
        "token_count": count_tokens(content),
    }


def _get_overlap_blocks(
    blocks: list[str],
    overlap_tokens: int,
) -> list[str]:
    """Select trailing blocks to reuse in the next chunk.

    Whole Markdown blocks are reused. The function stops before the desired
    overlap would be exceeded.

    Args:
        blocks: Blocks from the previous chunk.
        overlap_tokens: Maximum desired number of overlapping tokens.

    Returns:
        The trailing blocks selected for overlap.
    """
    if overlap_tokens == 0:
        return []

    overlap_blocks: list[str] = []
    token_count = 0

    for block in reversed(blocks):
        block_tokens = count_tokens(block)

        if token_count + block_tokens > overlap_tokens:
            break

        overlap_blocks.insert(0, block)
        token_count += block_tokens

    return overlap_blocks