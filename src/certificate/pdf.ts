/**
 * A one-page PDF that shows one JPEG over the whole page: the certificate download. Written by hand
 * (six objects and a cross-reference table, PDF 1.4) so the site needs no PDF library; the browser's
 * canvas makes the JPEG, which PDF embeds as is (DCTDecode).
 */

export interface PdfInfo {
  title: string;
  author: string;
  subject: string;
}

/** A4 landscape, in points. */
export const A4_LANDSCAPE = { width: 841.89, height: 595.28 };

export function jpegPdf(
  jpeg: Uint8Array,
  pixels: { width: number; height: number },
  info: PdfInfo,
  page: { width: number; height: number } = A4_LANDSCAPE,
): Uint8Array {
  const parts: Uint8Array[] = [];
  const offsets: number[] = [];
  let length = 0;
  const push = (chunk: string | Uint8Array) => {
    const bytes = typeof chunk === 'string' ? latin1(chunk) : chunk;
    parts.push(bytes);
    length += bytes.length;
  };
  const object = (n: number, ...body: (string | Uint8Array)[]) => {
    offsets[n] = length;
    push(`${n} 0 obj\n`);
    body.forEach(push);
    push('\nendobj\n');
  };

  const w = num(page.width);
  const h = num(page.height);
  const draw = `q ${w} 0 0 ${h} 0 0 cm /Im0 Do Q`;
  push('%PDF-1.4\n%\xe2\xe3\xcf\xd3\n'); // the binary comment marks the file as binary
  object(1, '<< /Type /Catalog /Pages 2 0 R >>');
  object(2, '<< /Type /Pages /Kids [3 0 R] /Count 1 >>');
  object(3, `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${w} ${h}] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>`);
  object(
    4,
    `<< /Type /XObject /Subtype /Image /Width ${pixels.width} /Height ${pixels.height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpeg.length} >>\nstream\n`,
    jpeg,
    '\nendstream',
  );
  object(5, `<< /Length ${draw.length} >>\nstream\n${draw}\nendstream`);
  object(6, `<< /Title ${text(info.title)} /Author ${text(info.author)} /Subject ${text(info.subject)} /Creator ${text('Relational Transformer Explained')} >>`);

  const xref = length;
  push(`xref\n0 7\n0000000000 65535 f \n`);
  for (let n = 1; n <= 6; n++) push(`${String(offsets[n]).padStart(10, '0')} 00000 n \n`);
  push(`trailer\n<< /Size 7 /Root 1 0 R /Info 6 0 R >>\nstartxref\n${xref}\n%%EOF\n`);

  const out = new Uint8Array(length);
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

/** A PDF text string in UTF-16BE (hex, with its byte-order mark), so any name survives. */
function text(s: string): string {
  let hex = 'FEFF';
  for (let i = 0; i < s.length; i++) hex += s.charCodeAt(i).toString(16).padStart(4, '0').toUpperCase();
  return `<${hex}>`;
}

function num(n: number): string {
  return String(Math.round(n * 100) / 100);
}

/** One byte per char (the PDF syntax here is ASCII plus the four bytes of the binary comment). */
function latin1(s: string): Uint8Array {
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 0xff;
  return out;
}
