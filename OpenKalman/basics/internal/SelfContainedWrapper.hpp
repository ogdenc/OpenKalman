/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for \ref internal::SelfContainedWrapper "SelfContainedWrapper"
 */

#ifndef OPENKALMAN_SELFCONTAINEDWRAPPER_HPP
#define OPENKALMAN_SELFCONTAINEDWRAPPER_HPP


namespace OpenKalman::internal
{
#ifdef __cpp_concepts
  template<indexible NestedObject, typename...Parameters>
#else
  template<typename NestedObject, typename...Parameters>
#endif
  struct SelfContainedWrapper : internal::library_base_t<SelfContainedWrapper<NestedObject, Parameters...>, NestedObject>
  {
  private:

    template<typename T>
    struct is_SelfContainedWrapper : std::false_type {};

    template<typename N, typename...Ps>
    struct is_SelfContainedWrapper<SelfContainedWrapper<N, Ps...>> : std::true_type {};

    static_assert(not is_SelfContainedWrapper<NestedObject>::value);

  public:

    /**
     * \brief Construct from a set of parameters, some of which may be stored in this object to extend their lifetimes.
     * \param p_tup A tuple of instances of Parameters, which will be internalized
     * \tparam Args Arguments to the constructor of BaseObject, which may include references to elements of p_tup.
     */
    template<typename...Args>
    explicit SelfContainedWrapper(std::tuple<Parameters...>&& p_tup, Args&&...args)
      : internalized_parameters {std::move(p_tup)}, nested {std::forward<Args>(args)...} {}


    /**
     * \brief Move constructor.
     */
    SelfContainedWrapper(SelfContainedWrapper&& arg) noexcept = default;


    /**
     * \brief Copy constructor. (Does not copy the internalized parameters.)
     */
    SelfContainedWrapper(const SelfContainedWrapper& arg)
      : internalized_parameters {}, nested {arg.nested} {}


    /**
     * \brief Move assignment operator.
     */
    SelfContainedWrapper& operator=(SelfContainedWrapper&& arg) = default;


    /**
     * \brief Copy assignment operator. (Does not copy the internalized parameters.)
     */
    SelfContainedWrapper& operator=(const SelfContainedWrapper& arg)
    {
      nested = arg.nested;
      internalized_parameters.reset();
    }


    /**
     * \brief Get the nested object.
     */
    [[nodiscard]] NestedObject& nested_object() & noexcept { return nested; }

    /// \overload
    [[nodiscard]] const NestedObject& nested_object() const & noexcept { return nested; }

    /// \overload
    [[nodiscard]] NestedObject&& nested_object() && noexcept { return std::move(*this).nested; }

    /// \overload
    [[nodiscard]] const NestedObject&& nested_object() const && noexcept { return std::move(*this).nested; }


    /**
     * \brief Convert to the nested object
     */
    operator NestedObject() & noexcept { return nested; }

    /// \overload
    operator NestedObject() const & noexcept { return nested; }

    /// \overload
    operator NestedObject() && noexcept { return std::move(*this).nested; }

    /// \overload
    operator NestedObject() const && noexcept { return std::move(*this).nested; }

  private:

    std::optional<std::tuple<Parameters...>> internalized_parameters;

    NestedObject nested;

  };


} // namespace OpenKalman::internal


#endif //OPENKALMAN_SELFCONTAINEDWRAPPER_HPP