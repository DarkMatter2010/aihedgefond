"""Synchronous, deterministic in-process command and event bus."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar, cast

from aihedgefund.core.schemas import Command, Event

CommandT = TypeVar("CommandT", bound=Command)
EventT = TypeVar("EventT", bound=Event)
CommandHandler = Callable[[CommandT], None]
EventHandler = Callable[[EventT], None]


class MessageBus(ABC):
    """Port for separate command and event message channels."""

    @abstractmethod
    def subscribe_command(
        self,
        command_type: type[CommandT],
        handler: CommandHandler[CommandT],
    ) -> None:
        """Register a command handler."""

    @abstractmethod
    def subscribe_event(
        self,
        event_type: type[EventT],
        handler: EventHandler[EventT],
    ) -> None:
        """Register an event subscriber."""

    @abstractmethod
    def publish_command(self, command: Command) -> None:
        """Dispatch a command to command handlers only."""

    @abstractmethod
    def publish_event(self, event: Event) -> None:
        """Dispatch an event to event subscribers only."""


class InProcessMessageBus(MessageBus):
    """Registration-ordered, synchronous message dispatcher."""

    def __init__(self) -> None:
        self._command_handlers: dict[type[Command], list[CommandHandler[Command]]] = defaultdict(
            list
        )
        self._event_subscribers: dict[type[Event], list[EventHandler[Event]]] = defaultdict(list)

    def subscribe_command(
        self,
        command_type: type[CommandT],
        handler: CommandHandler[CommandT],
    ) -> None:
        """Register a command handler in dispatch order."""
        if not issubclass(command_type, Command):
            msg = "command_type must inherit Command"
            raise TypeError(msg)
        self._command_handlers[command_type].append(
            cast(CommandHandler[Command], handler)
        )

    def subscribe_event(
        self,
        event_type: type[EventT],
        handler: EventHandler[EventT],
    ) -> None:
        """Register an event subscriber in dispatch order."""
        if not issubclass(event_type, Event):
            msg = "event_type must inherit Event"
            raise TypeError(msg)
        self._event_subscribers[event_type].append(cast(EventHandler[Event], handler))

    def publish_command(self, command: Command) -> None:
        """Synchronously dispatch a command to a stable handler snapshot."""
        if not isinstance(command, Command):
            msg = "publish_command accepts Command instances only"
            raise TypeError(msg)
        for handler in tuple(self._command_handlers[type(command)]):
            handler(command)

    def publish_event(self, event: Event) -> None:
        """Synchronously dispatch an event to a stable subscriber snapshot."""
        if not isinstance(event, Event):
            msg = "publish_event accepts Event instances only"
            raise TypeError(msg)
        for subscriber in tuple(self._event_subscribers[type(event)]):
            subscriber(event)
