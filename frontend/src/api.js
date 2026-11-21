const BASE_URL = import.meta.env.VITE_API_URL || 'https://ecosort-ai-vhmr.onrender.com'
async function request(path, options) {
  const response = await fetch(`${BASE_URL}${path}`, options)
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`)
  return body
}
export const api = {
  dashboard: () => request('/api/dashboard'),
  history: () => request('/api/predictions'),
  classify: (file, explain) => { const form = new FormData(); form.append('image', file); return request(`/api/classify?include_gradcam=${explain}`, { method: 'POST', body: form }) }
}
