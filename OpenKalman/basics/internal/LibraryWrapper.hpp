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

#include<iostream>

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief A dumb wrapper for \ref indexible objects so that they are treated exactly as native objects within a library.
   * \tparam NestedObject An indexible object that may or may not be in a library of interest.
   * \tparam LibraryObject Any object from the library to which this wrapper is to be associated.
   * \tparam InternalizedParameters If this LibraryWrapper is not otherwise self-contained, this are a full set of
   * arguments necessary to construct the object, which will be stored internally.
   */
#ifdef __cpp_concepts
  template<indexible NestedObject, indexible LibraryObject, typename...InternalizedParameters> requires
    (... and (not std::is_reference_v<InternalizedParameters>))
#else
  template<typename NestedObject, typename LibraryObject, typename...InternalizedParameters>
#endif
  struct LibraryWrapper : internal::library_base_t<LibraryWrapper<NestedObject, LibraryObject, InternalizedParameters...>, LibraryObject>
  {
  private:

    using Base = internal::library_base_t<LibraryWrapper, LibraryObject>;

#ifndef __cpp_concepts
    static_assert((... and (not std::is_reference_v<InternalizedParameters>)));
#endif

  public:

    /**
     * \brief Construct from a self-contained object.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<NestedObject> Arg> requires (not std::same_as<std::decay_t<Arg>, LibraryWrapper>) and
      (sizeof...(InternalizedParameters) == 0)
#else
    template<typename Arg, typename...Ps, std::enable_if_t<std::is_convertible_v<Arg, NestedObject> and
      (not std::is_same_v<std::decay_t<Arg>, LibraryWrapper>) and (sizeof...(InternalizedParameters) == 0), int> = 0>
#endif
    explicit LibraryWrapper(Arg&& arg) : wrapped_expression {std::forward<Arg>(arg)} {}


    /**
     * \overload
     * \brief Construct from a set of parameters, all of which will be stored in this object to extend their lifetime.
     * \tparam Ps Parameters from which to construct the nested object.
     */
#ifdef __cpp_concepts
    template<typename...Ps> requires (sizeof...(Ps) != 1 or (... and (not std::same_as<std::decay_t<Ps>, LibraryWrapper>)))
#else
    template<typename...Ps, std::enable_if_t<
      (sizeof...(Ps) != 1 or (... and (not std::is_same_v<std::decay_t<Ps>, LibraryWrapper>))), int> = 0>
#endif
    explicit LibraryWrapper(Ps&&...ps) :
      internalized_parameters {std::forward<Ps>(ps)...},
      wrapped_expression {std::apply([](auto&&...p){ return NestedObject {std::forward<decltype(p)>(p)...}; }, internalized_parameters)} {}


    /**
     * \brief Move constructor.
     */
    //LibraryWrapper(LibraryWrapper&& arg) = default;


    /**
     * \brief Copy constructor when there are no internalized parameters.
     */
#ifdef __cpp_concepts
    LibraryWrapper(const LibraryWrapper& arg) requires (sizeof...(InternalizedParameters) == 0)
#else
    template<bool params = (sizeof...(InternalizedParameters) == 0), std::enable_if_t<params, int> = 0>
    LibraryWrapper(const LibraryWrapper& arg)
#endif
      = default;


    /**
     * \overload
     * \brief Copy constructor when there are internalized parameters.
     */
#ifdef __cpp_concepts
    LibraryWrapper(const LibraryWrapper& arg) requires (sizeof...(InternalizedParameters) > 0)
#else
    template<bool params = (sizeof...(InternalizedParameters) > 0), std::enable_if_t<params, int> = 0>
    LibraryWrapper(const LibraryWrapper& arg)
#endif
    : internalized_parameters {arg.internalized_parameters},
      wrapped_expression {std::apply([](auto&&...p){ return NestedObject {std::forward<decltype(p)>(p)...}; }, internalized_parameters)} {}


    /**
     * \brief Assign from another compatible indexible object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires (sizeof...(InternalizedParameters) == 0) and
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>, decltype(to_native_matrix<NestedObject>(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<(sizeof...(InternalizedParameters) == 0) and
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
    LibraryWrapper& operator=(LibraryWrapper&& arg) = default;


    /**
     * \brief Copy assignment operator.
     */
    LibraryWrapper& operator=(const LibraryWrapper& arg)
    {
      internalized_parameters = arg.internalized_parameters;
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

    std::tuple<InternalizedParameters...> internalized_parameters;

    NestedObject wrapped_expression;

  };


} // namespace OpenKalman::internal


#endif //OPENKALMAN_LIBRARYWRAPPER_HPP