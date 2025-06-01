import requests

def main():
    try:
        base_url = "https://www.space-track.org"
        auth_path = "/ajaxauth/login"
        user_name = "hqjfnz.space.operations.centre@nzdf.mil.nz"  # Replace with your actual username
        password = "N3bula-Chugg6-1"         # Replace with your actual password
        query = "/basicspacedata/query/class/gp/orderby/NORAD_CAT_ID/format/3le"

        session = requests.Session()
        auth_response = session.post(base_url + auth_path, data={"identity": user_name, "password": password})

        if auth_response.status_code != 200:
            raise Exception("Authentication failed. Status code: {}".format(auth_response.status_code))

        response = session.get(base_url + query)
        
        if response.status_code == 200:
            with open("my_satellites.txt", "a") as file:  # Open file in append mode
                file.write(response.text)
        else:
            raise Exception("Failed to retrieve TLE data. Status code: {}".format(response.status_code))

        logout_response = session.get(base_url + "/ajaxauth/logout")
        if logout_response.status_code != 200:
            print("Logout failed. Status code: {}".format(logout_response.status_code))

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()