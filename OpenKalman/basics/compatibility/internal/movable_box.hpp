/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Header file for compatibility definition equivalent to the exposition-only class movable-box.
 */

#ifndef OPENKALMAN_COMPATIBILITY_MOVABLE_BOX_HPP
#define OPENKALMAN_COMPATIBILITY_MOVABLE_BOX_HPP

#include <optional>
#include <type_traits>
#include "basics/compatibility/language-features.hpp"

namespace OpenKalman::internal
{
  namespace detail
  {
    template<typename T>
#ifdef __cpp_lib_concepts
    concept boxable = std::copy_constructible<T> and std::is_object_v<T>;
#else
    inline constexpr bool boxable = copy_constructible<T> and std::is_object_v<T>;
#endif


    template<typename T>
#ifdef __cpp_lib_concepts
    concept boxable_copyable =
      std::copy_constructible<T> and (std::movable<T> or std::is_nothrow_move_constructible_v<T>);
#else
    inline constexpr bool boxable_copyable =
      copy_constructible<T> and
      (copyable<T> or (std::is_nothrow_move_constructible_v<T> or std::is_nothrow_copy_constructible_v<T>));
#endif


    template<typename T>
#ifdef __cpp_lib_concepts
    concept boxable_movable =
      (not std::copy_constructible<T>) and (std::movable<T> or std::is_nothrow_move_constructible_v<T>);
#else
    inline constexpr bool boxable_movable =
      (not copy_constructible<T>) and (movable<T> or std::is_nothrow_move_constructible_v<T>);
#endif
  }


#ifdef __cpp_lib_concepts
  template<detail::boxable>
#else
  template<typename T, typename = void>
#endif
  struct movable_box;


#ifdef __cpp_lib_concepts
  template<detail::boxable T>
  struct movable_box
#else
  template<typename T>
  struct movable_box<T, std::enable_if_t<detail::boxable<T> and
    not detail::boxable_movable<T> and not detail::boxable_copyable<T>>>
#endif
    : std::optional<T>
  {
    using std::optional<T>::optional;


#ifdef __cpp_lib_concepts
    constexpr
    movable_box() noexcept(std::is_nothrow_default_constructible_v<T>) requires std::default_initializable<T>
#else
    template<bool Enable = true, std::enable_if_t<Enable and std::is_default_constructible_v<T>, int> = 0>
    constexpr
    movable_box() noexcept(std::is_nothrow_default_constructible_v<T>)
#endif
    : std::optional<T>{std::in_place} {}


    movable_box(const movable_box&) = default;


    movable_box(movable_box&&) = default;


    using std::optional<T>::operator=;

    
#ifdef __cpp_lib_concepts
    constexpr movable_box&
    operator=(const movable_box& that) noexcept(std::is_nothrow_copy_constructible_v<T>)
      requires (not std::copyable<T>) && std::copy_constructible<T>
#else
    template<bool Enable = true, std::enable_if_t<Enable and (not copyable<T>) && copy_constructible<T>, int> = 0>
    constexpr movable_box&
    operator=(const movable_box& that) noexcept(std::is_nothrow_copy_constructible_v<T>)
#endif
    {
      if (this != std::addressof(that))
      {
        if ((bool) that) this->emplace(*that);
        else this->reset();
      }
      return *this;
    }

    
#ifdef __cpp_lib_concepts
    constexpr movable_box&
    operator=(movable_box&& that) noexcept(std::is_nothrow_move_constructible_v<T>) requires (not std::movable<T>)
#else
    template<bool Enable = true, std::enable_if_t<Enable and (not movable<T>), int> = 0>
    constexpr movable_box&
    operator=(movable_box&& that) noexcept(std::is_nothrow_move_constructible_v<T>)
#endif
    {
      if (this != std::addressof(that))
      {
        if ((bool) that) this->emplace(std::move(*that));
        else this->reset();
      }
      return *this;
    }
  };


#ifdef __cpp_lib_concepts
  template<detail::boxable T> requires detail::boxable_movable<T> or detail::boxable_copyable<T>
  struct movable_box<T>
#else
  template<typename T>
  struct movable_box<T, std::enable_if_t<detail::boxable<T> and (detail::boxable_movable<T> or detail::boxable_copyable<T>)>>
#endif
  {
  private:
    
    [[no_unique_address]] T M_value = T();

  public:

#ifdef __cpp_lib_concepts
    movable_box() requires std::default_initializable<T> = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and std::is_default_constructible_v<T>, int> = 0>
    movable_box() {};
#endif


#ifdef __cpp_lib_concepts
    constexpr explicit
    movable_box(const T& t) noexcept(std::is_nothrow_copy_constructible_v<T>) requires std::copy_constructible<T>
#else
    template<bool Enable = true, std::enable_if_t<Enable and copy_constructible<T>, int> = 0>
    constexpr explicit
    movable_box(const T& t) noexcept(std::is_nothrow_copy_constructible_v<T>)
#endif
      : M_value(t) {}


    constexpr explicit
    movable_box(T&& t) noexcept(std::is_nothrow_move_constructible_v<T>) : M_value(std::move(t)) {}


#ifdef __cpp_lib_concepts
    template<typename...Args> requires std::constructible_from<T, Args...>
#else
    template<typename...Args, std::enable_if_t<constructible_from<T, Args...>, int> = 0>
#endif
    constexpr explicit
    movable_box(std::in_place_t, Args&&...args) noexcept(std::is_nothrow_constructible_v<T, Args...>)
      : M_value(std::forward<Args>(args)...) {}


    movable_box(const movable_box&) = default;

    movable_box(movable_box&&) = default;

#ifdef __cpp_lib_concepts
    movable_box& operator=(const movable_box&) requires std::copyable<T> = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and copyable<T>, int> = 0>
    constexpr movable_box& operator=(const movable_box&) {};
#endif


#ifdef __cpp_lib_concepts
    movable_box& operator=(movable_box&&) requires std::movable<T> = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and movable<T>, int> = 0>
    constexpr movable_box& operator=(movable_box&&) {};
#endif


#ifdef __cpp_lib_concepts
    constexpr movable_box&
    operator=(const movable_box& that) noexcept requires (not std::copyable<T>) and std::copy_constructible<T>
#else
    template<bool Enable = true, std::enable_if_t<Enable and (not copyable<T>) and copy_constructible<T>, int> = 0>
    constexpr movable_box&
    operator=(const movable_box& that) noexcept
#endif
    {
      static_assert(std::is_nothrow_copy_constructible_v<T>);
      if (this != std::addressof(that))
      {
        M_value.~T();
#if __cplusplus >= 202002L
        std::construct_at(std::addressof(M_value), *that);
#else
      if constexpr (std::is_array_v<T>) return ::new (static_cast<void*>(std::addressof(M_value))) T[1]();
      else return ::new (static_cast<void*>(std::addressof(M_value))) T(*that);
#endif
      }
      return *this;
    }

    // Likewise for move assignment.
#ifdef __cpp_lib_concepts
    constexpr movable_box&
    operator=(movable_box&& that) noexcept requires (not std::movable<T>)
#else
    template<bool Enable = true, std::enable_if_t<Enable and (not movable<T>), int> = 0>
    constexpr movable_box&
    operator=(movable_box&& that) noexcept
#endif
    {
      static_assert(std::is_nothrow_move_constructible_v<T>);
      if (this != std::addressof(that))
      {
        M_value.~T();
#if __cplusplus >= 202002L
        std::construct_at(std::addressof(M_value), *that);
#else
        if constexpr (std::is_array_v<T>) return ::new (static_cast<void*>(std::addressof(M_value))) T[1]();
        else return ::new (static_cast<void*>(std::addressof(M_value))) T(std::move(*that));
#endif
      }
      return *this;
    }

    constexpr bool
    has_value() const noexcept { return true; };

    constexpr T&
    operator*() & noexcept { return M_value; }

    constexpr const T&
    operator*() const & noexcept { return M_value; }

    constexpr T&&
    operator*() && noexcept { return std::move(M_value); }

    constexpr const T&&
    operator*() const && noexcept { return std::move(M_value); }

    constexpr T *
    operator->() noexcept { return std::addressof(M_value); }

    constexpr const T *
    operator->() const noexcept { return std::__addressof(M_value); }

  };

}

#endif //OPENKALMAN_COMPATIBILITY_MOVABLE_BOX_HPP
