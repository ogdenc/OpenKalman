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
#include "basics/compatibility/core-concepts.hpp"

namespace OpenKalman::internal
{
  namespace detail
  {
    template<typename T>
#ifdef __cpp_lib_concepts
    concept boxable =
#else
    inline constexpr bool boxable =
#endif
      stdcompat::move_constructible<T> and std::is_object_v<T>;


    template<typename Tp>
#ifdef __cpp_lib_concepts
    concept boxable_copyable =
#else
    inline constexpr bool boxable_copyable =
#endif
      stdcompat::copy_constructible<Tp> and
      (stdcompat::copyable<Tp> or (std::is_nothrow_move_constructible_v<Tp> and std::is_nothrow_copy_constructible_v<Tp>));


    template<typename Tp>
#ifdef __cpp_lib_concepts
    concept boxable_movable =
#else
    inline constexpr bool boxable_movable =
#endif
      (not stdcompat::copy_constructible<Tp>) and
      (stdcompat::movable<Tp> or std::is_nothrow_move_constructible_v<Tp>);

  }


  /**
   * \brief Equivalent to exposition-only "movable-box" from c++23 standard
   */
#ifdef __cpp_lib_concepts
  template<detail::boxable T>
  struct movable_box : std::optional<T>
#else
  template<typename T, typename = void>
  struct movable_box;

  template<typename T>
  struct movable_box<T, std::enable_if_t<
    detail::boxable<T> and not detail::boxable_movable<T> and not detail::boxable_copyable<T>>> : std::optional<T>
#endif
  {
    using std::optional<T>::optional;


#ifdef __cpp_lib_concepts
    constexpr
    movable_box() noexcept(std::is_nothrow_default_constructible_v<T>) requires std::default_initializable<T>
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::default_initializable<T>, int> = 0>
    constexpr
    movable_box() noexcept(std::is_nothrow_default_constructible_v<T>)
#endif
    : std::optional<T>{std::in_place} {}


    constexpr
    movable_box(const movable_box&) = default;


    constexpr
    movable_box(movable_box&&) = default;


    using std::optional<T>::operator=;

    
#ifdef __cpp_lib_concepts
    constexpr movable_box&
    operator=(const movable_box& that) noexcept(std::is_nothrow_copy_constructible_v<T>)
      requires (not std::copyable<T>) && std::copy_constructible<T>
#else
    template<bool Enable = true, std::enable_if_t<Enable and (not stdcompat::copyable<T>) && stdcompat::copy_constructible<T>, int> = 0>
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
    template<bool Enable = true, std::enable_if_t<Enable and (not stdcompat::movable<T>), int> = 0>
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
    constexpr
    movable_box() requires std::default_initializable<T> = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::default_initializable<T>, int> = 0>
    constexpr
    movable_box() {};
#endif


#ifdef __cpp_lib_concepts
    constexpr explicit
    movable_box(const T& t) noexcept(std::is_nothrow_copy_constructible_v<T>) requires std::copy_constructible<T>
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::copy_constructible<T>, int> = 0>
    constexpr explicit
    movable_box(const T& t) noexcept(std::is_nothrow_copy_constructible_v<T>)
#endif
      : M_value(t) {}


    constexpr explicit
    movable_box(T&& t) noexcept(std::is_nothrow_move_constructible_v<T>) : M_value(std::move(t)) {}


#ifdef __cpp_lib_concepts
    template<typename...Args> requires std::constructible_from<T, Args...>
#else
    template<typename...Args, std::enable_if_t<stdcompat::constructible_from<T, Args...>, int> = 0>
#endif
    constexpr explicit
    movable_box(std::in_place_t, Args&&...args) noexcept(std::is_nothrow_constructible_v<T, Args...>)
      : M_value(std::forward<Args>(args)...) {}


    movable_box(const movable_box&) = default;

    movable_box(movable_box&&) = default;


#ifdef __cpp_lib_concepts
    movable_box& operator=(const movable_box&) requires std::copyable<T> = default;


    constexpr movable_box&
    operator=(const movable_box& that) noexcept requires (not std::copyable<T>) and std::copy_constructible<T>
    {
      static_assert(std::is_nothrow_copy_constructible_v<T>);
      if (this != std::addressof(that))
      {
        M_value.~T();
        std::construct_at(std::addressof(M_value), *that);
      }
      return *this;
    }
#else
    constexpr movable_box& operator=(const movable_box& that)
    {
      static_assert(stdcompat::copy_constructible<T> and std::is_nothrow_copy_constructible_v<T>);
      if constexpr (stdcompat::copyable<T>)
      if (this != std::addressof(that))
      {
        if constexpr (stdcompat::copyable<T>)
        {
          M_value = that.M_value;
        }
        else
        {
          M_value.~T();
          if constexpr (std::is_array_v<T>) return ::new (static_cast<void*>(std::addressof(M_value))) T[1]();
          else return ::new (static_cast<void*>(std::addressof(M_value))) T(*that);
        }
      }
      return *this;
    };
#endif


#ifdef __cpp_lib_concepts
    movable_box& operator=(movable_box&&) requires std::movable<T> = default;


    constexpr movable_box&
    operator=(movable_box&& that) noexcept requires (not std::movable<T>)
    {
      static_assert(std::is_nothrow_move_constructible_v<T>);
      if (this != std::addressof(that))
      {
        M_value.~T();
        std::construct_at(std::addressof(M_value), *that);
      }
      return *this;
    }
#else
    constexpr movable_box& operator=(movable_box&& that) noexcept
    {
      static_assert(std::is_nothrow_move_constructible_v<T>);
      if (this != std::addressof(that))
      {
        if constexpr (stdcompat::movable<T>)
        {
          M_value = std::move(that.M_value);
        }
        else
        {
          M_value.~T();
          if constexpr (std::is_array_v<T>) return ::new (static_cast<void*>(std::addressof(M_value))) T[1]();
          else return ::new (static_cast<void*>(std::addressof(M_value))) T(std::move(*that));
        }
      }
      return *this;
    }
#endif


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
    operator->() const noexcept { return std::addressof(M_value); }

  };

}

#endif
