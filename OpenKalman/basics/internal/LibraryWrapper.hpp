/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for internal::LibraryWrapper
 */

#ifndef OPENKALMAN_LIBRARYWRAPPER_HPP
#define OPENKALMAN_LIBRARYWRAPPER_HPP


namespace OpenKalman::internal
{

#ifdef __cpp_concepts
  template<indexible NestedObject, indexible LibraryObject>
#else
  template<typename NestedObject, typename LibraryObject>
#endif
  struct LibraryWrapper : internal::library_base_t<LibraryWrapper<NestedObject, LibraryObject>, LibraryObject>
  {
  private:

    using Base = internal::library_base_t<LibraryWrapper, LibraryObject>;

  public:

    /**
     * \brief Construct from a non-library object.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<NestedObject> Arg> requires (not std::same_as<std::decay_t<Arg>, LibraryWrapper>)
#else
    template<typename Arg, typename...Ps, std::enable_if_t<std::is_convertible_v<Arg, NestedObject> and
      (not std::is_same_v<std::decay_t<Arg>, LibraryWrapper>), int> = 0>
#endif
    explicit LibraryWrapper(Arg&& arg) : wrapped_expression {std::forward<Arg>(arg)} {}


    /**
     * \brief Move constructor.
     */
    LibraryWrapper(LibraryWrapper&& arg) = default;


    /**
     * \brief Copy constructor.
     */
    LibraryWrapper(const LibraryWrapper& arg) = default;


    /**
     * \brief Assign from another compatible indexible object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires (not std::same_as<std::decay_t<Arg>, LibraryWrapper>) and
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>, decltype(to_native_matrix<NestedObject>(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<(not std::same_as<std::decay_t<Arg>, LibraryWrapper>) and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, decltype(to_native_matrix<NestedObject>(std::declval<Arg&&>()))>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      wrapped_expression = to_native_matrix<NestedObject>(std::forward<Arg>(arg));
      return *this;
    }


    /**
     * \brief Move assignment operator.
     */
    LibraryWrapper& operator=(LibraryWrapper&& arg)
    {
      if (this != &arg) wrapped_expression = std::move(arg);
      return *this;
    }


    /**
     * \brief Copy assignment operator.
     */
    LibraryWrapper& operator=(const LibraryWrapper& arg)
    {
      if (this != &arg) wrapped_expression = arg;
      return *this;
    }


    /**
     * \brief Get the nested object.
     */
    auto& nested_object() & noexcept { return wrapped_expression; }

    /// \overload
    const auto& nested_object() const & noexcept { return wrapped_expression; }

    /// \overload
    auto&& nested_object() && noexcept { return std::move(*this).wrapped_expression; }

    /// \overload
    const auto&& nested_object() const && noexcept { return std::move(*this).wrapped_expression; }

  private:

    NestedObject wrapped_expression;

  };


} // namespace OpenKalman::internal


#endif //OPENKALMAN_LIBRARYWRAPPER_HPP