export function csrfToken(){return document.cookie.split("; ").find((v)=>v.startsWith("customer_service_csrf="))?.split("=")[1] ?? "";}
