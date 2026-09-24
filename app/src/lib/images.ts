import type { ImageAttachment, ImageMediaType } from '../types/chat'

// Keep in sync with backend/upscale/schemas.py
export const ACCEPTED_IMAGE_TYPES: ImageMediaType[] = ['image/png', 'image/jpeg', 'image/webp', 'image/gif']
export const MAX_IMAGE_BYTES = 5 * 1024 * 1024
export const MAX_ATTACHMENTS = 4

export function toDataUrl(image: ImageAttachment): string {
  return `data:${image.media_type};base64,${image.data}`
}

export function readImageFile(file: File): Promise<ImageAttachment> {
  return new Promise((resolve, reject) => {
    if (!ACCEPTED_IMAGE_TYPES.includes(file.type as ImageMediaType)) {
      reject(new Error(`${file.name}: only PNG, JPEG, WebP or GIF images are supported.`))
      return
    }
    if (file.size > MAX_IMAGE_BYTES) {
      reject(new Error(`${file.name}: images must be 5 MB or smaller.`))
      return
    }
    const reader = new FileReader()
    reader.onload = () => {
      const dataUrl = reader.result as string
      resolve({
        name: file.name,
        media_type: file.type as ImageMediaType,
        data: dataUrl.slice(dataUrl.indexOf(',') + 1),
      })
    }
    reader.onerror = () => reject(new Error(`${file.name}: could not read file.`))
    reader.readAsDataURL(file)
  })
}

const EXTENSIONS: Record<ImageMediaType, string> = {
  'image/png': 'png',
  'image/jpeg': 'jpg',
  'image/webp': 'webp',
  'image/gif': 'gif',
}

function pastedFileName(type: string, index: number, count: number): string {
  const stamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '-')
  const ext = EXTENSIONS[type as ImageMediaType] ?? type.slice('image/'.length)
  const suffix = count > 1 ? `-${index + 1}` : ''
  return `pasted-${stamp}${suffix}.${ext}`
}

function renamePasted(files: File[]): File[] {
  return files.map((file, i) => new File([file], pastedFileName(file.type, i, files.length), { type: file.type }))
}

/**
 * Image files from a clipboard paste, renamed so each gets a readable, unique name
 * (browsers name every pasted screenshot "image.png"). Returns [] for text pastes so
 * they keep their default behavior — including Office copies, which carry rich text
 * alongside a rendered image. Screenshot tools that also put a plain-text file path
 * on the clipboard still paste as an image.
 */
export function imagesFromClipboard(data: DataTransfer): File[] {
  const hasText = data.getData('text/plain').trim().length > 0
  const hasRichText = Array.from(data.types).some((t) => t === 'text/html' || t === 'text/rtf')
  if (hasText && hasRichText) return []

  let files = Array.from(data.items)
    .filter((item) => item.kind === 'file' && item.type.startsWith('image/'))
    .flatMap((item) => item.getAsFile() ?? [])
  // WebKitGTK (Linux/WSLg) may expose a pasted image only through `files`.
  if (files.length === 0) files = Array.from(data.files).filter((f) => f.type.startsWith('image/'))
  return renamePasted(files)
}

/**
 * Reads images with the async Clipboard API. Used when the paste event carried no data
 * (WebKitGTK doesn't always expose clipboard images to it). Throws if access is denied.
 */
export async function readClipboardImages(): Promise<File[]> {
  const blobs: Blob[] = []
  for (const item of await navigator.clipboard.read()) {
    const type = item.types.find((t) => t.startsWith('image/'))
    if (type) blobs.push(await item.getType(type))
  }
  return renamePasted(blobs.map((b) => new File([b], 'image', { type: b.type })))
}

/**
 * Converts images in formats the backend doesn't accept (e.g. the image/bmp that
 * Windows screenshots can arrive as through WSLg) to PNG. Other files pass through.
 */
export async function toSupportedImage(file: File): Promise<File> {
  if (ACCEPTED_IMAGE_TYPES.includes(file.type as ImageMediaType) || !file.type.startsWith('image/')) return file
  const bitmap = await createImageBitmap(file)
  const canvas = document.createElement('canvas')
  canvas.width = bitmap.width
  canvas.height = bitmap.height
  canvas.getContext('2d')!.drawImage(bitmap, 0, 0)
  bitmap.close()
  const png = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error(`${file.name}: could not convert image.`))), 'image/png'),
  )
  return new File([png], file.name.replace(/\.[^.]*$/, '') + '.png', { type: 'image/png' })
}
