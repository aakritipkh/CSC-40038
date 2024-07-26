import React from "react";
import {
  BrowserRouter as Router,
  Route,
  Routes
} from "react-router-dom";
import Home from "./pages/home";
import Results from "./pages/results";

export default function BasicExample() {
  return (
    <Router>
      <div>
        {/* Navigation to another route via link */}
        {/* <ul>
          <li>
            <Link to="/">Home</Link>
          </li>
        </ul> */}
        <Routes>
          <Route exact path="/" element={<Home/>} />
          <Route exact path="/results" element={<Results/>} />
        </Routes>
      </div>
    </Router>
  );
}
