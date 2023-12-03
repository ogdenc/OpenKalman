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
  /**
   * \internal
   * \brief A dumb wrapper for \ref indexible objects so that they are treated exactly as native objects within a library.
   * \tparam NestedObject An indexible object that may or may not be in a library of interest.
   * \tparam LibraryObject Any object from the library to which this wrapper is to be associated.
   */
#ifdef __cpp_concepts
  template<indexible NestedObject, indexible LibraryObject>
#else
  template<typename NestedObject, typename LibraryObject = NestedObject>
#endif
  struct LibraryWrapper : internal::library_base_t<LibraryWrapper<NestedObject, LibraryObject>, LibraryObject>
  {
  private:

    using Base = internal::library_base_t<LibraryWrapper, LibraryObject>;

  public:

#ifdef __cpp_concepts
    template<typename Arg> requires (not std::same_as<std::decay_t<Arg>, LibraryWrapper>) and
    std::constructible_from<NestedObject, Arg&&>
#else
    template<typename Arg, std::enable_if_t<(not std::is_same_v<std::decay_t<Arg>, LibraryWrapper>) and
      std::is_constructible_v<NestedObject, Arg&&>, int> = 0>
#endif
    explicit LibraryWrapper(Arg&& arg) : wrapped_expression {std::forward<Arg>(arg)} {}

  private:

    template<typename Arg, typename Indices>
    constexpr void
    init(const Arg& arg, Indices& indices, std::size_t ix)
    {
      auto dim = get_index_dimension_of(wrapped_expression, ix);
      if (ix < indices.size())
      {
        for (std::size_t i = 0; i < dim; ++i)
        {
          indices[ix] == i;
          init(arg, indices, ix + 1);
        }
      }
      else set_element(wrapped_expression, get_element(std::forward<Arg>(arg), indices), indices);
    }

  public:

#ifdef __cpp_concepts
    template<typename Arg> requires (not std::same_as<std::decay_t<Arg>, LibraryWrapper>) and
    (not std::constructible_from<NestedObject, Arg&&>) and std::default_initializable<NestedObject> and
      writable<NestedObject>
#else
    template<typename Arg, std::enable_if_t<(not std::is_same_v<std::decay_t<Arg>, LibraryWrapper>) and
      (not std::is_constructible_v<NestedObject, Arg&&>) and std::is_default_constructible_v<NestedObject> and
      writable<NestedObject>, int> = 0>
#endif
    explicit LibraryWrapper(const Arg& arg) : wrapped_expression{}
    {
      std::vector<std::size_t> vec {get_index_count(arg)};
      init(arg, vec, 0);
    }


    /**
     * \brief Assign from another compatible indexible object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>, decltype(to_native_matrix<NestedObject>(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<
      std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, decltype(to_native_matrix<NestedObject>(std::declval<Arg&&>()))>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      wrapped_expression = to_native_matrix<NestedObject>(std::forward<Arg>(arg));
      return *this;
    }


    /**
     * \brief Get the nested matrix.
     */
    decltype(auto) nested_matrix() & noexcept { return (wrapped_expression); }

    /// \overload
    decltype(auto) nested_matrix() const & noexcept { return (wrapped_expression); }

    /// \overload
    decltype(auto) nested_matrix() && noexcept { return (std::move(*this).wrapped_expression); }

    /// \overload
    decltype(auto) nested_matrix() const && noexcept { return (std::move(*this).wrapped_expression); }

  protected:

    NestedObject wrapped_expression;

  };


} // namespace OpenKalman::internal


#endif //OPENKALMAN_LIBRARYWRAPPER_HPP