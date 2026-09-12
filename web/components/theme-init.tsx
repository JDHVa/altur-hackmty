export function ThemeInit() {
  const js = `try{if(localStorage.getItem("centinela-theme")==="light")document.documentElement.classList.remove("dark")}catch(e){}`;
  return <script dangerouslySetInnerHTML={{ __html: js }} />;
}
