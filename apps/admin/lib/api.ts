import {csrfToken} from "./auth";
const base=process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
export async function api<T>(path:string, init:RequestInit={}) : Promise<T>{const method=init.method??"GET";const response=await fetch(`${base}${path}`,{...init,credentials:"include",headers:{"Content-Type":"application/json",...(method!=="GET"?{"X-CSRF-Token":csrfToken()}:{}),...init.headers}});if(!response.ok)throw new Error(await response.text());return response.json();}
export type StaffSession = {sub:string;email:string};
export const session=()=>api<StaffSession>("/admin/auth/session");
export const cases=()=>api<any[]>("/admin/cases"); export const detail=(id:string)=>api<any>(`/admin/cases/${id}`); export const act=(id:string,action:string,version:number,body={})=>api<any>(`/admin/cases/${id}/${action}`,{method:"POST",body:JSON.stringify({version,...body})});
