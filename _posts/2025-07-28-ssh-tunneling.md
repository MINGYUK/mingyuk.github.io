---
layout: post
title: SSH 터널링을 이용한 내부망 접근
date: 2025-07-28 00:01:00
description: SSH 터널링을 이용해 원격 데스크톱을 대체해보자.
tags:
  - 한국어
categories: remote-work
pseudocode: false
toc:
  sidebar: true
featured: false
tabs: false
---

# 왜 쓴 글인가?

요즘 R studio server 등 연구실 내에서 돌리는 서비스를 외부에서 사용하기 위해 다들 다양한 방법을 쓰고 있는 것 같다. 그 중에서도 구글 원격 데스크톱처럼 화면을 전송하는 솔루션을 많이들 쓰는 것 같은데, 당연히 전송량이 많아 느리고 압축에 의해 화면이 깨지기도 하는 등 여러모로 많이 불편하다. 심지어 내부 정책에 의해 원격 데스크톱 프로그램이 막힐 때마다 옮겨다니는 유목민도 있는 것으로 보인다. 트래픽도 적고 빠르고 안정적인, 직접 브라우저로 서비스에 접근할 수 있는 방법이 있는데도 말이다. 혹시 본인이 이런 상황이라면 참고해보자. 작게는 NAS에 접근할 때도 사용할 수 있고, 주피터 서버 등등 다양한 상황에 활용할 수 있다.

# SSH 터널

일단 외부에서 SSH로 접속할 수 있는 호스트가 있긴 있어야 한다. 본인의 연구실 컴퓨터에 SSH server를 활성화하거나, 혹시 다같이 공유하는 jump host가 있으면 이를 통하는 것이 가장 편할듯. 기본적으로 내부 서비스와 통하는 트래픽을 이 ssh 호스트를 통하게 하는 방법이다. 보통 ssh로 접속하려면,

```console
ssh user@host-ip
```

와 같이 접속하는데, 이때 예를 들어 내부 주소 server-A, 포트 8080에서 서비스가 돌아간다고 하자. 그럼 원래 연구실 안에서 쓸 때는 브라우저에 server-A:8080이라고 쳤을 것이다. 외부에서 같은 방법으로는 당연히 내부 주소니까 접속이 불가능하다. 그래서 SSH 터널을 연결하여, localhost로 server-A의 내부주소를 대체하는 것이다.

```console
ssh -L 1234:server-A:8080 user@host-ip
```

이렇게 호스트에 접속하면, 호스트가 터널의 역할을 하게 된다. 접속한 쪽에서 localhost:1234로 오고 가는 트래픽이 실제로는 호스트를 통해 server-A:8080으로 전달되는 것이다. 그러면 SSH 연결을 한 뒤에 브라우저에서는 localhost:1234에 접속하면 원래 쓰던 서비스를 그대로 사용할 수 있다.

# 혹시 내부 주소를 사용하는 설정이 존재하는 경우

우리 연구실의 경우, 내부 주소를 hard-coded 방식으로 사용하는 설정이 있었다. 위의 가정을 그대로 들면 server-A의 포트 9090에서 돌아가는 다른 API를 8080에서 돌아가는 서비스가 server-A 내부 주소를 그대로 사용해서 호출하는 방식이었다. 그러니까 위의 방식처럼 8080 포트만 터널링을 하면, remote machine에서 그대로 그 내부 주소를 사용해서 API를 호출하므로 작동하지 않는다. 이 경우 두가지 작업이 필요하다.

일단 server-A 내부 주소를 remote machine에서 localhost로 바꿔서 부르게 해야 한다. 도메인 주소가 설정된 경우, /etc/hosts 파일에 `127.0.0.1 domain-name` 한줄을 추가해두면 도메인 이름이 localhost로 resolve된다. 도메인 이름이 없이 실제 ip 주소만을 사용하는 경우, iptables를 직접 사용해야 한다. 이때 iptables 설정은 재부팅을 하면 초기화되므로 쉘 스크립트를 작성해두거나, iptables-persistent 등 다른 방법을 사용해야 한다. 나는 개인적으로 굳이 영구적으로 resolve할 필요도 없고 해서, 쉘 스크립트를 작성했다.

```shell
#!/bin/bash

# Define the target IP address and the redirect IP (localhost)
TARGET_IP="some-ip-address"
REDIRECT_IP="127.0.0.1"

# Function to add the iptables rule
add_rule() {
    echo "Adding iptables rule: Redirecting traffic for $TARGET_IP to $REDIRECT_IP (localhost)..."
    # Add the DNAT rule to the OUTPUT chain of the nat table
    # This rule applies to packets originating from your local machine
    sudo iptables -t nat -A OUTPUT -d "$TARGET_IP" -j DNAT --to-destination "$REDIRECT_IP"
    if [ $? -eq 0 ]; then
        echo "Rule added successfully."
        echo "You can verify with: sudo iptables -t nat -L OUTPUT -v -n"
    else
        echo "Failed to add the rule. Check your permissions and iptables setup."
    fi
}

# Function to remove the iptables rule
remove_rule() {
    echo "Removing iptables rule: Redirecting traffic for $TARGET_IP to $REDIRECT_IP (localhost)..."
    # Delete the DNAT rule from the OUTPUT chain of the nat table
    sudo iptables -t nat -D OUTPUT -d "$TARGET_IP" -j DNAT --to-destination "$REDIRECT_IP"
    if [ $? -eq 0 ]; then
        echo "Rule removed successfully."
    else
        echo "Failed to remove the rule. It might not exist or there was an error."
    fi
}

# Check if the script is run with sudo
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run with sudo. Please run: sudo ./$(basename "$0") [add|remove]"
    exit 1
fi

# Main logic based on arguments
case "$1" in
    add)
        add_rule
        ;;
    remove)
        remove_rule
        ;;
    *)
        echo "Usage: sudo ./$(basename "$0") [add|remove]"
        echo "  add    : Adds the iptables redirection rule."
        echo "  remove : Removes the iptables redirection rule."
        exit 1
        ;;
esac
```

chmod +x를 이용해 실행할 수 있는 스크립트로 만들고, ./script.sh add와 같이 실행하면 이제 현재 환경에서 server-A 주소로 오고가는 트래픽은 실제로는 localhost로 보내진다.

이후 SSH 접속 시에 포트를 하나 더 터널링해야 한다.

```console
ssh -L 1234:server-A:8080 -L 9090:server-A:9090 user@host
```

이와 같이 하면 정상적으로 remote client에서 localhost:1234에 접속해서 원래 사용하던 서비스를 사용할 수 있다. (두번째 포트는 hard-coded여서 포트 숫자도 똑같이 맞춰줘야 했음)

# 맺음

별안간 원격 데스크톱으로 스스로의 혈압을 높이고 있는 원격 근무자들에게 도움이 되길. 웹 서비스 뿐만 아니라 내부망에서 사용되는 서비스들에 대부분 사용할 수 있다. SSH 터널링이라는 개념에 대해 잘 알아두자.
